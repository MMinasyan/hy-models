import torch
import torch.nn.functional as F
import math
from flash_attn import flash_attn_func, flash_attn_varlen_func


def prepare_4d_attn_mask(padding_mask=None, q_padding_mask=None, causal=False, windows_size=None, src_seq_len=None, tgt_seq_len=None, dtype=torch.float32, device=None):
    """
    Prepare attention mask by combining causal and padding masks if provided.

    Args:
        padding_mask (torch.Tensor, optional): Padding mask tensor of shape (batch_size, src_seq_len).
            Float tensor of 0s and 1s, where 0s are padding positions.
        causal (bool, optional): Whether to use causal masking. Defaults to False.
        window_size (int, optional): Window size for sliding window masking. Defaults to None.
            If causal is True, (-window_size, 0) are window positions.
            If causal is False, windows(-window_size, window_size) are window positions.
        seq_len (int, optional): Sequence length. Defaults to None. If padding_mask is provided, seq_len is inferred from padding_mask.size(1).
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.
        device (torch.device, optional): Device. Defaults to None. If padding_mask is provided, device is inferred from padding_mask.device.

    Returns:
        Attention mask tensor of shape (batch_size, 1, tgt_seq_len, src_seq_len) or None if none of causal, padding_mask and windows_size are provided.
    
    For self-attention: tgt_seq_len = src_seq_len = seq_len
    For cross-attention: only padding_mask is used, shape (batch_size, 1, 1, src_seq_len)
    """
    if not causal and padding_mask is None and q_padding_mask is None and windows_size is None:
        return None
    
    if device is None and (padding_mask is not None or q_padding_mask is not None):
        device = padding_mask.device if padding_mask is not None else q_padding_mask.device
    
    if src_seq_len is None:
        if padding_mask is None:
            raise ValueError("src_seq_len must be provided if padding_mask is not provided")
        src_seq_len = padding_mask.size(1)
    
    if tgt_seq_len is None:
        if q_padding_mask is None:
            raise ValueError("tgt_seq_len must be provided if q_padding_mask is not provided")     
        tgt_seq_len = q_padding_mask.size(1)
    
    # Get min value for dtype to use as -inf
    min_dtype = float('-inf') # torch.finfo(dtype).min
    attn_mask = None
    
    # Handle self-attention case (causal or window masking)
    if causal or windows_size is not None:
        if tgt_seq_len != src_seq_len:
            raise ValueError(f"causal and window masking is not supported for cross-attention, expected tgt_seq_len and src_seq_len to be the same, got tgt_seq_len: {tgt_seq_len} and src_seq_len: {src_seq_len}")
        
        # Causal mask: -inf for future positions (j > i)
        if causal:
            causal_mask = torch.triu(torch.full((tgt_seq_len, tgt_seq_len), min_dtype, device=device), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            attn_mask = causal_mask
        
        # Window mask
        if windows_size is not None:
            relative_positions = torch.arange(tgt_seq_len, device=device).unsqueeze(0) - \
                               torch.arange(tgt_seq_len, device=device).unsqueeze(1)
            if causal:
                # Causal window: i - windows_size <= j <= i
                window_mask = torch.where(
                    (relative_positions <= 0) & (relative_positions >= -windows_size),
                    0.0, min_dtype
                )
            else:
                # Non-causal window: |j - i| <= windows_size
                window_mask = torch.where(
                    torch.abs(relative_positions) <= windows_size,
                    0.0, min_dtype
                )
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            attn_mask = window_mask if attn_mask is None else attn_mask + window_mask

    if padding_mask is not None or q_padding_mask is not None:
        batch_size = padding_mask.size(0) if padding_mask is not None else q_padding_mask.size(0)
        combined_padding_mask = torch.zeros(
        (batch_size, 1, tgt_seq_len, src_seq_len),
        device=padding_mask.device if padding_mask is not None else q_padding_mask.device,
        dtype=dtype  # Assuming min_dtype is a float type like torch.float32
        )
        # Handle key/value padding mask
        if padding_mask is not None:
            key_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_seq_len]
            key_padding_mask = torch.where(key_padding_mask == 0, min_dtype, 0.0)
            combined_padding_mask = combined_padding_mask + key_padding_mask

        # Handle query padding mask
        if q_padding_mask is not None:
            query_padding_mask = q_padding_mask.unsqueeze(1).unsqueeze(3)  # [batch_size, 1, tgt_seq_len, 1]
            query_padding_mask = torch.where(query_padding_mask == 0, min_dtype, 0.0)
            combined_padding_mask = combined_padding_mask + query_padding_mask

        if attn_mask is None:
            attn_mask = combined_padding_mask # Only padding mask
        else:
            attn_mask = attn_mask + combined_padding_mask # combine for self-attention
    
    return attn_mask.to(dtype) if attn_mask is not None else None


def _eager_attention_forward(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=None,
        scale=None,
        output_attentions=False,
):
    scale = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale # [batch_size, num_heads, tgt_seq_len, src_seq_len]

    if attn_mask is not None:
        # print('attn_mask.shape:', attn_mask.shape)
        # print('attn_scores.shape:', attn_scores.shape)
        min_val = torch.finfo(attn_mask.dtype).min
        attn_mask = torch.where(torch.isneginf(attn_mask), min_val, attn_mask)
        attn_scores = attn_scores + attn_mask

    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)

    if dropout_p is not None:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    output = torch.matmul(attn_weights, value) # [batch_size, num_heads, tgt_seq_len, head_dim]

    if output_attentions:
        return output, attn_weights
    else:
        return output


def eager_attn_forward(query, key, value, causal=False, padding_mask=None, q_padding_mask=None, window_size=None, output_attentions=False, dropout_p=0.0):
    """
    Eager attention forward pass.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, tgt_seq_len, num_heads, head_dim)
        key (torch.Tensor): Key tensor of shape (batch_size, src_seq_len, num_heads, head_dim)
        value (torch.Tensor): Value tensor of shape (batch_size, src_seq_len, num_heads, head_dim)
        causal (bool, optional): Whether to use causal attention. Defaults to False.
        padding_mask (torch.Tensor, optional): Padding mask tensor of shape (batch_size, src_seq_len).
            Float tensor of 0s and 1s, where 0s are padding positions.
        q_padding_mask (torch.Tensor, optional): Padding mask tensor of shape (batch_size, tgt_seq_len).
            Float tensor of 0s and 1s, where 0s are padding positions.
        window_size (int, optional): Window size for sliding window attention. Defaults to None.
            (-window_size, 0) positions are unmasked when causal is True, (-window_size, window_size) when causal is False.
        output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.

    Returns:
        Output tensor of shape (batch_size, tgt_seq_len, num_heads, head_dim)
    """
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_mask = prepare_4d_attn_mask(
        padding_mask=padding_mask,
        q_padding_mask=q_padding_mask,
        causal=causal,
        windows_size=window_size,
        src_seq_len=key.size(2) if padding_mask is None else None,
        tgt_seq_len=query.size(2) if q_padding_mask is None else None,
        dtype=query.dtype,
        device=query.device
    )

    return _eager_attention_forward(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, output_attentions=output_attentions).transpose(1, 2)


def sdpa_attn_forward(query, key, value, causal=False, padding_mask=None, q_padding_mask=None, window_size=None, dropout_p=0.0):
    """
    SDPA attention forward pass.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, tgt_seq_len, num_heads, head_dim)
        key (torch.Tensor): Key tensor of shape (batch_size, src_seq_len, num_heads, head_dim)
        value (torch.Tensor): Value tensor of shape (batch_size, src_seq_len, num_heads, head_dim)
        causal (bool, optional): Whether to use causal attention. Defaults to False.
        padding_mask (torch.Tensor, optional): Padding mask tensor of shape (batch_size, src_seq_len).
            Float tensor of 0s and 1s, where 0s are padding positions in key/value inputs.
        q_padding_mask (torch.Tensor, optional): Padding mask tensor of shape (batch_size, tgt_seq_len).
            Float tensor of 0s and 1s, where 0s are padding positions in query inputs.
        window_size (int, optional): Window size for sliding window attention. Defaults to None.
            (-window_size, 0) positions are unmasked when causal is True, (-window_size, window_size) when causal is False.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.

    Returns:
        Output tensor of shape (batch_size, tgt_seq_len, num_heads, head_dim)
    """
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_mask = prepare_4d_attn_mask(
        padding_mask=padding_mask,
        q_padding_mask=q_padding_mask,
        causal=causal,
        windows_size=window_size,
        src_seq_len=key.size(2) if padding_mask is None else None,
        tgt_seq_len=query.size(2) if q_padding_mask is None else None,
        dtype=query.dtype,
        device=query.device
    )
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p).transpose(1, 2)


def _get_flash_attn_unpad_data(mask):
    """
    Compute indices, cumulative sequence lengths, and max sequence length from a mask.
    
    Args:
        mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len) where True indicates valid tokens.
    
    Returns:
        tuple: (indices, cu_seqlens, max_seqlen_in_batch)
    """
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cumsum_seqlens = seqlens_in_batch.cumsum(dim=0).to(torch.int32)
    cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32, device=mask.device), cumsum_seqlens], dim=0)
    return indices, cu_seqlens, max_seqlen_in_batch


def flash_attn_forward(query, key, value, causal=False, padding_mask=None, q_padding_mask=None, window_size=None, dropout_p=0.0, dtype=torch.bfloat16):
    """
    FlashAttention forward pass.
    If padding_mask or q_padding_mask is provided, flash_attn_varlen_func will be used.
    Otherwise, flash_attn_func will be used.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, tgt_seq_len, num_heads, head_dim)
        key (torch.Tensor): Key tensor of shape (batch_size, src_seq_len, num_heads, head_dim)
        value (torch.Tensor): Value tensor of shape (batch_size, src_seq_len, num_heads, head_dim)
        causal (bool, optional): Whether to use causal attention. Defaults to False.
        padding_mask (torch.Tensor, optional): Padding mask tensor of shape (batch_size, src_seq_len).
            Float tensor of 0s and 1s, where 0s are padding positions. flash_attn_varlen_func will be used if none of padding_mask or q_padding_mask is provided.
        q_padding_mask (torch.Tensor, optional): Padding mask tensor of shape (batch_size, tgt_seq_len).
            Float tensor of 0s and 1s, where 0s are padding positions. flash_attn_varlen_func will be used if none of padding_mask or q_padding_mask is provided.
        window_size (int, optional): Window size for sliding window attention. Defaults to None.
            (-window_size, 0) positions are unmasked when causal is True, (-window_size, window_size) when causal is False.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        dtype (torch.dtype, optional): Data type. Defaults to torch.bfloat16.
    """
    if window_size is not None:
        if causal:
            window_size = (window_size, 0)
        else:
            window_size = (window_size, window_size)
    else:
        window_size = (-1, -1)
    if padding_mask is None and q_padding_mask is None:
        return flash_attn_func(
            query.to(dtype) if query.dtype != dtype else query,
            key.to(dtype) if key.dtype != dtype else key,
            value.to(dtype) if value.dtype != dtype else value,
            causal=causal,
            window_size=window_size,
            dropout_p=dropout_p
            )
    else:
        batch_size, tgt_seq_len, num_heads, head_dim = query.shape
        src_seq_len = key.shape[1]

        # Process query
        if q_padding_mask is not None:
            indices_q, cu_seqlens_q, max_seqlen_q = _get_flash_attn_unpad_data(q_padding_mask)
            query_flat = query.reshape(batch_size * tgt_seq_len, num_heads, head_dim)[indices_q]
        else:
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * tgt_seq_len, step=tgt_seq_len, 
                                    dtype=torch.int32, device=query.device)
            max_seqlen_q = tgt_seq_len
            query_flat = query.reshape(batch_size * tgt_seq_len, num_heads, head_dim)
        
        # Process key and value
        if padding_mask is not None:
            indices_k, cu_seqlens_k, max_seqlen_k = _get_flash_attn_unpad_data(padding_mask)
            key_flat = key.reshape(batch_size * src_seq_len, num_heads, head_dim)[indices_k]
            value_flat = value.reshape(batch_size * src_seq_len, num_heads, head_dim)[indices_k]
        else:
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * src_seq_len, step=src_seq_len, 
                                    dtype=torch.int32, device=key.device)
            max_seqlen_k = src_seq_len
            key_flat = key.reshape(batch_size * src_seq_len, num_heads, head_dim)
            value_flat = value.reshape(batch_size * src_seq_len, num_heads, head_dim)

        # Compute attention using flash_attn_varlen_func
        attn_output_unpad = flash_attn_varlen_func(
            query_flat.to(dtype) if query_flat.dtype != dtype else query_flat,
            key_flat.to(dtype) if key_flat.dtype != dtype else key_flat,
            value_flat.to(dtype) if value_flat.dtype != dtype else value_flat,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size=window_size,
            dropout_p=dropout_p,
        )
        # Pad output back to original shape if query was unpadded
        if indices_q is not None:
            attn_output = torch.zeros(batch_size * tgt_seq_len, num_heads, head_dim, 
                                    dtype=attn_output_unpad.dtype, device=attn_output_unpad.device)
            attn_output[indices_q] = attn_output_unpad
            attn_output = attn_output.view(batch_size, tgt_seq_len, num_heads, head_dim)
        else:
            attn_output = attn_output_unpad.view(batch_size, tgt_seq_len, num_heads, head_dim)
        
        return attn_output
