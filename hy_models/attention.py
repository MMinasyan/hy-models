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
        return_attn_weights=False,
):
    """
    Eager attention forward pass.
    Args:
        query (torch.Tensor): tensor of shape [batch_size, num_heads, tgt_seq_len, head_dim]
        key (torch.Tensor): tensor of shape [batch_size, num_heads, src_seq_len, head_dim]
        value (torch.Tensor): tensor of shape [batch_size, num_heads, src_seq_len, head_dim]
        attn_mask (torch.Tensor, optional): A float tensor of shape [1, 1, tgt_seq_len, src_seq_len]
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        scale (float, optional): Scale factor for the attention scores. Defaults to None.
        return_attn_weights (bool, optional): Whether to return the attention weights. Defaults to False.
    """
    scale = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale # [batch_size, num_heads, tgt_seq_len, src_seq_len]

    if attn_mask is not None:
        attn_scores = attn_scores + attn_mask

    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype) # for stability during mixed precision training

    if dropout_p is not None:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    output = torch.matmul(attn_weights, value) # [batch_size, num_heads, tgt_seq_len, head_dim]

    if return_attn_weights:
        return output, attn_weights
    else:
        return output
    

