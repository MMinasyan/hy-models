import torch
import torch.nn.functional as F
import math


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


