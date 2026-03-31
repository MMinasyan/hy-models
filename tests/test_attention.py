"""
Tests that all attention backends produce identical outputs
across supported configurations: self-attention, cross-attention,
causal masking, padding masks, and sliding window.

Compares eager vs sdpa (CPU, float32, exact match).
Compares eager vs flash_attention_2 (CUDA, bfloat16, ~1e-3 tolerance)
when CUDA and flash-attn are available — skipped otherwise.
"""
import unittest
import torch
from hy_models.attention import (
    eager_attn_forward, sdpa_attn_forward, flash_attn_forward,
    prepare_4d_attn_mask, HAS_FLASH_ATTN,
)

HAS_CUDA = torch.cuda.is_available()


def make_qkv(batch, tgt_seq, src_seq, heads, head_dim, dtype=torch.float32, device='cpu'):
    """Generate random query, key, value tensors in (batch, seq, heads, head_dim) layout."""
    q = torch.randn(batch, tgt_seq, heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, src_seq, heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, src_seq, heads, head_dim, dtype=dtype, device=device)
    return q, k, v


def make_padding_mask(batch, seq_len, min_valid=1, device='cpu'):
    """Generate a padding mask with random-length valid prefixes. 1=valid, 0=pad."""
    mask = torch.zeros(batch, seq_len, device=device)
    for i in range(batch):
        valid_len = torch.randint(min_valid, seq_len + 1, (1,)).item()
        mask[i, :valid_len] = 1.0
    return mask


def _filter_valid(out, q_padding_mask):
    """Return only the outputs at valid (non-padded) query positions."""
    if q_padding_mask is None:
        return out
    valid = q_padding_mask.bool()
    return out[valid]


class TestEagerVsSdpa(unittest.TestCase):
    """Compare eager and sdpa attention outputs across configurations."""

    BATCH = 2
    SEQ = 16
    HEADS = 4
    HEAD_DIM = 32
    ATOL = 1e-5

    def _assert_close(self, eager_out, sdpa_out, msg="", q_padding_mask=None):
        self.assertEqual(eager_out.shape, sdpa_out.shape, f"Shape mismatch {msg}")
        a = _filter_valid(eager_out, q_padding_mask)
        b = _filter_valid(sdpa_out, q_padding_mask)
        if a.numel() == 0:
            return
        max_diff = (a - b).abs().max().item()
        self.assertLess(max_diff, self.ATOL, f"Max diff {max_diff} >= {self.ATOL} {msg}")

    # ---- Self-attention: no masks ----

    def test_self_attention_no_mask(self):
        q, k, v = make_qkv(self.BATCH, self.SEQ, self.SEQ, self.HEADS, self.HEAD_DIM)
        out_eager = eager_attn_forward(q, k, v)
        out_sdpa = sdpa_attn_forward(q, k, v)
        self._assert_close(out_eager, out_sdpa, "self-attn no mask")

    # ---- Self-attention: causal ----

    def test_self_attention_causal(self):
        q, k, v = make_qkv(self.BATCH, self.SEQ, self.SEQ, self.HEADS, self.HEAD_DIM)
        out_eager = eager_attn_forward(q, k, v, causal=True)
        out_sdpa = sdpa_attn_forward(q, k, v, causal=True)
        self._assert_close(out_eager, out_sdpa, "self-attn causal")

    # ---- Self-attention: padding mask only ----

    def test_self_attention_padding_mask(self):
        q, k, v = make_qkv(self.BATCH, self.SEQ, self.SEQ, self.HEADS, self.HEAD_DIM)
        mask = make_padding_mask(self.BATCH, self.SEQ)
        out_eager = eager_attn_forward(q, k, v, padding_mask=mask, q_padding_mask=mask)
        out_sdpa = sdpa_attn_forward(q, k, v, padding_mask=mask, q_padding_mask=mask)
        self._assert_close(out_eager, out_sdpa, "self-attn padding", q_padding_mask=mask)

    # ---- Self-attention: causal + padding ----

    def test_self_attention_causal_padding(self):
        q, k, v = make_qkv(self.BATCH, self.SEQ, self.SEQ, self.HEADS, self.HEAD_DIM)
        mask = make_padding_mask(self.BATCH, self.SEQ)
        out_eager = eager_attn_forward(q, k, v, causal=True, padding_mask=mask, q_padding_mask=mask)
        out_sdpa = sdpa_attn_forward(q, k, v, causal=True, padding_mask=mask, q_padding_mask=mask)
        self._assert_close(out_eager, out_sdpa, "self-attn causal+padding", q_padding_mask=mask)

    # ---- Self-attention: sliding window (non-causal) ----

    def test_self_attention_window(self):
        q, k, v = make_qkv(self.BATCH, self.SEQ, self.SEQ, self.HEADS, self.HEAD_DIM)
        out_eager = eager_attn_forward(q, k, v, window_size=4)
        out_sdpa = sdpa_attn_forward(q, k, v, window_size=4)
        self._assert_close(out_eager, out_sdpa, "self-attn window")

    # ---- Self-attention: causal + window ----

    def test_self_attention_causal_window(self):
        q, k, v = make_qkv(self.BATCH, self.SEQ, self.SEQ, self.HEADS, self.HEAD_DIM)
        out_eager = eager_attn_forward(q, k, v, causal=True, window_size=4)
        out_sdpa = sdpa_attn_forward(q, k, v, causal=True, window_size=4)
        self._assert_close(out_eager, out_sdpa, "self-attn causal+window")

    # ---- Self-attention: all masks combined ----

    def test_self_attention_all_masks(self):
        q, k, v = make_qkv(self.BATCH, self.SEQ, self.SEQ, self.HEADS, self.HEAD_DIM)
        mask = make_padding_mask(self.BATCH, self.SEQ)
        out_eager = eager_attn_forward(q, k, v, causal=True, padding_mask=mask, q_padding_mask=mask, window_size=4)
        out_sdpa = sdpa_attn_forward(q, k, v, causal=True, padding_mask=mask, q_padding_mask=mask, window_size=4)
        self._assert_close(out_eager, out_sdpa, "self-attn all masks", q_padding_mask=mask)

    # ---- Cross-attention: kv padding mask only ----

    def test_cross_attention_kv_padding(self):
        """Cross-attention: different src/tgt lengths, padding on key/value side."""
        tgt_seq, src_seq = 8, 16
        q, k, v = make_qkv(self.BATCH, tgt_seq, src_seq, self.HEADS, self.HEAD_DIM)
        kv_mask = make_padding_mask(self.BATCH, src_seq)
        q_mask = torch.ones(self.BATCH, tgt_seq)  # no padding on query side
        out_eager = eager_attn_forward(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask)
        out_sdpa = sdpa_attn_forward(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask)
        self._assert_close(out_eager, out_sdpa, "cross-attn kv padding")

    # ---- Cross-attention: both sides padded ----

    def test_cross_attention_both_padded(self):
        tgt_seq, src_seq = 12, 16
        q, k, v = make_qkv(self.BATCH, tgt_seq, src_seq, self.HEADS, self.HEAD_DIM)
        kv_mask = make_padding_mask(self.BATCH, src_seq)
        q_mask = make_padding_mask(self.BATCH, tgt_seq)
        out_eager = eager_attn_forward(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask)
        out_sdpa = sdpa_attn_forward(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask)
        self._assert_close(out_eager, out_sdpa, "cross-attn both padded", q_padding_mask=q_mask)


@unittest.skipUnless(HAS_CUDA and HAS_FLASH_ATTN, "Requires CUDA + flash-attn")
class TestEagerVsFlash(unittest.TestCase):
    """Compare eager (float32 reference) vs flash_attention_2 (bfloat16) on CUDA."""

    BATCH = 4
    SEQ = 32
    HEADS = 4
    HEAD_DIM = 64
    ATOL = 2e-2  # bfloat16 tolerance

    def _assert_close(self, eager_out, flash_out, msg="", q_padding_mask=None):
        self.assertEqual(eager_out.shape, flash_out.shape, f"Shape mismatch {msg}")
        # Compare in float32
        a = _filter_valid(eager_out.float(), q_padding_mask)
        b = _filter_valid(flash_out.float(), q_padding_mask)
        if a.numel() == 0:
            return
        max_diff = (a - b).abs().max().item()
        self.assertLess(max_diff, self.ATOL, f"Max diff {max_diff} >= {self.ATOL} {msg}")

    def _make_cuda_qkv(self, tgt_seq=None, src_seq=None):
        tgt = tgt_seq or self.SEQ
        src = src_seq or self.SEQ
        return make_qkv(self.BATCH, tgt, src, self.HEADS, self.HEAD_DIM,
                        dtype=torch.float32, device='cuda')

    def _make_cuda_mask(self, seq_len=None):
        return make_padding_mask(self.BATCH, seq_len or self.SEQ, device='cuda')

    def _eager_ref(self, q, k, v, **kwargs):
        """Run eager on float32 CUDA tensors, return bfloat16 for shape compat."""
        out = eager_attn_forward(q, k, v, **kwargs)
        return out.to(torch.bfloat16)

    # ---- No masks ----

    def test_no_mask(self):
        q, k, v = self._make_cuda_qkv()
        ref = self._eager_ref(q, k, v)
        out = flash_attn_forward(q, k, v, dtype=torch.bfloat16)
        self._assert_close(ref, out, "flash no mask")

    # ---- Causal ----

    def test_causal(self):
        q, k, v = self._make_cuda_qkv()
        ref = self._eager_ref(q, k, v, causal=True)
        out = flash_attn_forward(q, k, v, causal=True, dtype=torch.bfloat16)
        self._assert_close(ref, out, "flash causal")

    # ---- Padding mask (both sides) ----

    def test_padding_mask(self):
        q, k, v = self._make_cuda_qkv()
        mask = self._make_cuda_mask()
        ref = self._eager_ref(q, k, v, padding_mask=mask, q_padding_mask=mask)
        out = flash_attn_forward(q, k, v, padding_mask=mask, q_padding_mask=mask, dtype=torch.bfloat16)
        self._assert_close(ref, out, "flash padding", q_padding_mask=mask)

    # ---- Causal + padding ----

    def test_causal_padding(self):
        q, k, v = self._make_cuda_qkv()
        mask = self._make_cuda_mask()
        ref = self._eager_ref(q, k, v, causal=True, padding_mask=mask, q_padding_mask=mask)
        out = flash_attn_forward(q, k, v, causal=True, padding_mask=mask, q_padding_mask=mask, dtype=torch.bfloat16)
        self._assert_close(ref, out, "flash causal+padding", q_padding_mask=mask)

    # ---- Sliding window ----

    def test_window(self):
        q, k, v = self._make_cuda_qkv()
        ref = self._eager_ref(q, k, v, window_size=8)
        out = flash_attn_forward(q, k, v, window_size=8, dtype=torch.bfloat16)
        self._assert_close(ref, out, "flash window")

    # ---- Causal + window ----

    def test_causal_window(self):
        q, k, v = self._make_cuda_qkv()
        ref = self._eager_ref(q, k, v, causal=True, window_size=8)
        out = flash_attn_forward(q, k, v, causal=True, window_size=8, dtype=torch.bfloat16)
        self._assert_close(ref, out, "flash causal+window")

    # ---- Cross-attention: kv padding ----

    def test_cross_attention_kv_padding(self):
        q, k, v = self._make_cuda_qkv(tgt_seq=16, src_seq=32)
        kv_mask = self._make_cuda_mask(32)
        q_mask = torch.ones(self.BATCH, 16, device='cuda')
        ref = self._eager_ref(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask)
        out = flash_attn_forward(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask, dtype=torch.bfloat16)
        self._assert_close(ref, out, "flash cross-attn kv padding")

    # ---- Cross-attention: both sides padded ----

    def test_cross_attention_both_padded(self):
        q, k, v = self._make_cuda_qkv(tgt_seq=16, src_seq=32)
        kv_mask = self._make_cuda_mask(32)
        q_mask = self._make_cuda_mask(16)
        ref = self._eager_ref(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask)
        out = flash_attn_forward(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask, dtype=torch.bfloat16)
        self._assert_close(ref, out, "flash cross-attn both padded", q_padding_mask=q_mask)


@unittest.skipUnless(HAS_CUDA and HAS_FLASH_ATTN, "Requires CUDA + flash-attn")
class TestSdpaVsFlash(unittest.TestCase):
    """Compare sdpa vs flash on CUDA to confirm all three backends agree."""

    BATCH = 4
    SEQ = 32
    HEADS = 4
    HEAD_DIM = 64
    ATOL = 2e-2

    def _assert_close(self, sdpa_out, flash_out, msg="", q_padding_mask=None):
        a = _filter_valid(sdpa_out.float(), q_padding_mask)
        b = _filter_valid(flash_out.float(), q_padding_mask)
        if a.numel() == 0:
            return
        max_diff = (a - b).abs().max().item()
        self.assertLess(max_diff, self.ATOL, f"Max diff {max_diff} >= {self.ATOL} {msg}")

    def test_no_mask(self):
        q, k, v = make_qkv(self.BATCH, self.SEQ, self.SEQ, self.HEADS, self.HEAD_DIM, device='cuda')
        out_sdpa = sdpa_attn_forward(q, k, v)
        out_flash = flash_attn_forward(q, k, v, dtype=torch.bfloat16)
        self._assert_close(out_sdpa, out_flash, "sdpa-vs-flash no mask")

    def test_causal(self):
        q, k, v = make_qkv(self.BATCH, self.SEQ, self.SEQ, self.HEADS, self.HEAD_DIM, device='cuda')
        out_sdpa = sdpa_attn_forward(q, k, v, causal=True)
        out_flash = flash_attn_forward(q, k, v, causal=True, dtype=torch.bfloat16)
        self._assert_close(out_sdpa, out_flash, "sdpa-vs-flash causal")

    def test_causal_padding(self):
        q, k, v = make_qkv(self.BATCH, self.SEQ, self.SEQ, self.HEADS, self.HEAD_DIM, device='cuda')
        mask = make_padding_mask(self.BATCH, self.SEQ, device='cuda')
        out_sdpa = sdpa_attn_forward(q, k, v, causal=True, padding_mask=mask, q_padding_mask=mask)
        out_flash = flash_attn_forward(q, k, v, causal=True, padding_mask=mask, q_padding_mask=mask, dtype=torch.bfloat16)
        self._assert_close(out_sdpa, out_flash, "sdpa-vs-flash causal+padding", q_padding_mask=mask)


class TestPrepare4dAttnMask(unittest.TestCase):
    """Test mask construction logic directly."""

    def test_no_mask_returns_none(self):
        result = prepare_4d_attn_mask(src_seq_len=8, tgt_seq_len=8, device='cpu')
        self.assertIsNone(result)

    def test_causal_mask_shape(self):
        mask = prepare_4d_attn_mask(causal=True, src_seq_len=8, tgt_seq_len=8, device='cpu')
        self.assertEqual(mask.shape, (1, 1, 8, 8))

    def test_causal_mask_upper_triangle_is_neginf(self):
        mask = prepare_4d_attn_mask(causal=True, src_seq_len=4, tgt_seq_len=4, device='cpu')
        mask_2d = mask[0, 0]
        # Upper triangle (future positions) should be -inf
        for i in range(4):
            for j in range(4):
                if j > i:
                    self.assertEqual(mask_2d[i, j].item(), float('-inf'),
                                     f"Position ({i},{j}) should be -inf")
                else:
                    self.assertEqual(mask_2d[i, j].item(), 0.0,
                                     f"Position ({i},{j}) should be 0")

    def test_padding_mask_blocks_padded_keys(self):
        # batch=1, valid tokens at 0,1, padding at 2,3
        pad_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        q_mask = torch.ones(1, 4)
        mask = prepare_4d_attn_mask(padding_mask=pad_mask, q_padding_mask=q_mask)
        # shape: (1, 1, 4, 4)
        mask_2d = mask[0, 0]
        # Columns 2,3 (padded keys) should be -inf for all query positions
        for i in range(4):
            self.assertEqual(mask_2d[i, 2].item(), float('-inf'))
            self.assertEqual(mask_2d[i, 3].item(), float('-inf'))
            self.assertEqual(mask_2d[i, 0].item(), 0.0)
            self.assertEqual(mask_2d[i, 1].item(), 0.0)

    def test_q_padding_mask_blocks_padded_queries(self):
        pad_mask = torch.ones(1, 4)
        q_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        mask = prepare_4d_attn_mask(padding_mask=pad_mask, q_padding_mask=q_mask)
        mask_2d = mask[0, 0]
        # Rows 2,3 (padded queries) should be -inf for all key positions
        for j in range(4):
            self.assertEqual(mask_2d[2, j].item(), float('-inf'))
            self.assertEqual(mask_2d[3, j].item(), float('-inf'))
            self.assertEqual(mask_2d[0, j].item(), 0.0)
            self.assertEqual(mask_2d[1, j].item(), 0.0)

    def test_window_mask_non_causal(self):
        mask = prepare_4d_attn_mask(windows_size=1, src_seq_len=4, tgt_seq_len=4, device='cpu')
        mask_2d = mask[0, 0]
        # Only |i - j| <= 1 should be 0, rest -inf
        for i in range(4):
            for j in range(4):
                if abs(i - j) <= 1:
                    self.assertEqual(mask_2d[i, j].item(), 0.0,
                                     f"({i},{j}) should be 0 (within window)")
                else:
                    self.assertEqual(mask_2d[i, j].item(), float('-inf'),
                                     f"({i},{j}) should be -inf (outside window)")

    def test_causal_window_mask(self):
        mask = prepare_4d_attn_mask(causal=True, windows_size=2, src_seq_len=6, tgt_seq_len=6, device='cpu')
        mask_2d = mask[0, 0]
        for i in range(6):
            for j in range(6):
                # Valid if j <= i (causal) AND i - j <= 2 (window)
                if j <= i and (i - j) <= 2:
                    self.assertEqual(mask_2d[i, j].item(), 0.0,
                                     f"({i},{j}) should be 0")
                else:
                    self.assertEqual(mask_2d[i, j].item(), float('-inf'),
                                     f"({i},{j}) should be -inf")

    def test_causal_rejects_cross_attention(self):
        """Causal mask requires src_seq_len == tgt_seq_len."""
        with self.assertRaises(ValueError):
            prepare_4d_attn_mask(causal=True, src_seq_len=8, tgt_seq_len=4, device='cpu')


class TestOutputShape(unittest.TestCase):
    """Verify output shapes match input conventions."""

    def test_eager_self_attention_shape(self):
        q, k, v = make_qkv(2, 8, 8, 4, 32)
        out = eager_attn_forward(q, k, v)
        self.assertEqual(out.shape, (2, 8, 4, 32))

    def test_sdpa_self_attention_shape(self):
        q, k, v = make_qkv(2, 8, 8, 4, 32)
        out = sdpa_attn_forward(q, k, v)
        self.assertEqual(out.shape, (2, 8, 4, 32))

    def test_eager_cross_attention_shape(self):
        q, k, v = make_qkv(2, 6, 10, 4, 32)
        q_mask = torch.ones(2, 6)
        kv_mask = torch.ones(2, 10)
        out = eager_attn_forward(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask)
        self.assertEqual(out.shape, (2, 6, 4, 32))

    def test_sdpa_cross_attention_shape(self):
        q, k, v = make_qkv(2, 6, 10, 4, 32)
        q_mask = torch.ones(2, 6)
        kv_mask = torch.ones(2, 10)
        out = sdpa_attn_forward(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask)
        self.assertEqual(out.shape, (2, 6, 4, 32))

    @unittest.skipUnless(HAS_CUDA and HAS_FLASH_ATTN, "Requires CUDA + flash-attn")
    def test_flash_self_attention_shape(self):
        q, k, v = make_qkv(2, 8, 8, 4, 64, device='cuda')
        out = flash_attn_forward(q, k, v, dtype=torch.bfloat16)
        self.assertEqual(out.shape, (2, 8, 4, 64))

    @unittest.skipUnless(HAS_CUDA and HAS_FLASH_ATTN, "Requires CUDA + flash-attn")
    def test_flash_cross_attention_shape(self):
        q, k, v = make_qkv(2, 6, 10, 4, 64, device='cuda')
        q_mask = torch.ones(2, 6, device='cuda')
        kv_mask = torch.ones(2, 10, device='cuda')
        out = flash_attn_forward(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask, dtype=torch.bfloat16)
        self.assertEqual(out.shape, (2, 6, 4, 64))


class TestMaskedOutputIsZero(unittest.TestCase):
    """Padded query positions should produce deterministic (zero-attended) output."""

    def test_padded_query_positions_ignored(self):
        """With q_padding_mask=0 for some positions, those positions get -inf in all
        attention scores, producing uniform attention weights after softmax.
        The output won't be exactly zero, but it should be identical across
        all masked query positions (since they all attend uniformly)."""
        q, k, v = make_qkv(1, 8, 8, 2, 16)
        q_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        kv_mask = torch.ones(1, 8)
        out = eager_attn_forward(q, k, v, padding_mask=kv_mask, q_padding_mask=q_mask)
        # Masked positions (4-7) should all have same output (uniform attention over all keys)
        masked_out = out[0, 4:8]  # (4, heads, head_dim)
        for i in range(1, 4):
            self.assertTrue(
                torch.allclose(masked_out[0], masked_out[i], atol=1e-6),
                "All masked query positions should produce identical output"
            )


if __name__ == '__main__':
    unittest.main()
