import torch
from torch import nn
import torch.nn.functional as F
import math


class Conv1dEmbedding(nn.Module):
    def __init__(self, embed_dim, hidden_size, vocab_size, inter_dim=512, kernel_size=5, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=inter_dim, kernel_size=kernel_size, stride=1, padding='same', bias=False)
        self.linear = nn.Linear(inter_dim * 2, hidden_size, bias=True)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, stride=3, padding=3, bias=True)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        embed_dim = self.embed.weight.size(-1)
        nn.init.normal_(self.embed.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.norm1(x)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Patch Merging
        batch_size, channels, seq_len = x.shape
        if seq_len % 2 != 0:
            x = F.pad(x, (0, 1))
            seq_len += 1
        x = x.view(batch_size, channels, seq_len // 2, 2)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, seq_len // 2, -1)

        x = self.linear(x)
        x = self.norm2(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)

        x = self.conv2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1)


class RotaryPositionalEmbeddings(nn.Module):
    """A PyTorch module that applies rotary positional embeddings to input tensors.

    Args:
        dim (int): The dimension per head (head_dim). Must be even.
        max_position (int): The maximum sequence length to precompute the embeddings for.
        base (float, optional): The base for the geometric progression of rotation angles. Defaults to 10000.
    """
    def __init__(self, dim: int, max_position: int = 4096, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.max_position = max_position
        self.base = base

        theta = 1.0 / (
            base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        pos = torch.arange(max_position, dtype=torch.float32).reshape(-1, 1, 1, 1)
        dim_t = theta.reshape(1, 1, 1, -1)
        angles = pos * dim_t
        rope_cache = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        self.register_buffer('rope_cache', rope_cache)

    def forward(self, x: torch.Tensor, position_ids = None) -> torch.Tensor:
        """Applies rotary positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_heads, dim).
            position_ids (torch.Tensor, optional): Tensor of position indices of shape (seq_len,).
                If None, defaults to torch.arange(seq_len).

        Returns:
            torch.Tensor: Output tensor with rotary embeddings applied, same shape as input.
        """
        batch_size, seq_len, num_heads, dim = x.shape
        assert dim == self.dim, "Input dimension mismatch"

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            assert position_ids.shape == (seq_len,), "position_ids should be of shape (seq_len,)"
            assert (position_ids >= 0).all() and (position_ids < self.max_position).all(), \
                "position_ids must be between 0 and max_position - 1"

        rope_cache = self.rope_cache[position_ids]
        cos_vals = rope_cache[..., 0]
        sin_vals = rope_cache[..., 1]

        cos_vals = cos_vals.view(1, seq_len, 1, dim // 2)
        sin_vals = sin_vals.view(1, seq_len, 1, dim // 2)

        x_reshaped = x.float().view(batch_size, seq_len, num_heads, dim // 2, 2)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]

        out_even = x_even * cos_vals - x_odd * sin_vals
        out_odd = x_even * sin_vals + x_odd * cos_vals

        out_reshaped = torch.stack([out_even, out_odd], dim=-1)
        out = out_reshaped.view(batch_size, seq_len, num_heads, dim)
        return out.type_as(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_dim: int,
        bias: bool = False,
        num_layers: int = 16
    ):
        super().__init__()
        self.bias = bias
        self.num_layers = num_layers

        self.w = nn.Linear(hidden_size, intermediate_dim, bias=bias)
        self.v = nn.Linear(hidden_size, intermediate_dim, bias=bias)
        self.out = nn.Linear(intermediate_dim, hidden_size, bias=bias)
        self.silu = nn.SiLU()

        self._init_weights()

    def _init_weights(self):
        std = 0.02 / math.sqrt(self.num_layers)
        nn.init.normal_(self.w.weight, mean=0.0, std=std)
        nn.init.normal_(self.v.weight, mean=0.0, std=std)
        nn.init.normal_(self.out.weight, mean=0.0, std=std)
        if self.bias == True:
            nn.init.constant_(self.w.bias, 0)
            nn.init.constant_(self.v.bias, 0)
            nn.init.constant_(self.out.bias, 0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.silu(self.w(x)) * self.v(x)
        return self.out(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, num_key_value_heads=None, bias=False, is_decoder=False, pos_encoding=None, num_layers=12):
        """
        Initialize the multi-headed self-attention layer with support for Grouped Query Attention.

        Args:
            embed_dim (int): Embedding dimension of the input.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            num_key_value_heads (int, optional): Number of query groups for Grouped Query Attention.
                If None, defaults to num_heads (standard MHA). Defaults to None.
            bias (bool, optional): If True, adds bias to input/output projection layers. Defaults to False.
            is_decoder (bool, optional): If True, applies causal attention (decoder mode).
                If False, applies full attention (encoder mode). Defaults to False.
            pos_encoding (nn.Module, optional): Positional encoding module. Defaults to None.
            num_layers (int, optional): Number of layers in the model, used for weight initialization. Defaults to 12.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.dropout = dropout
        self.bias = bias
        self.is_decoder = is_decoder  # New parameter to control causal attention
        self.num_layers = num_layers

        # Ensure num_heads is divisible by num_key_value_heads
        assert self.num_heads % self.num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_proj = nn.Linear(embed_dim, self.head_dim * self.num_key_value_heads, bias=bias)
        self.value_proj = nn.Linear(embed_dim, self.head_dim * self.num_key_value_heads, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = pos_encoding

        self._reset_parameters()

    def _reset_parameters(self):
        # std = 0.02
        # kv_std = std / (self.num_heads / self.num_key_value_heads) ** 0.5
        # out_std = std / (2 * self.num_layers)**0.5
        std = 0.02 / math.sqrt(self.num_layers)
        kv_std = std / (self.num_heads / self.num_key_value_heads) ** 0.5
        torch.nn.init.normal_(self.query_proj.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.key_proj.weight, mean=0.0, std=kv_std)
        torch.nn.init.normal_(self.value_proj.weight, mean=0.0, std=kv_std)
        torch.nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)

        # Handle biases if they exist (bias=True)
        if self.bias:
            torch.nn.init.constant_(self.query_proj.bias, 0.0)
            torch.nn.init.constant_(self.key_proj.bias, 0.0)
            torch.nn.init.constant_(self.value_proj.bias, 0.0)
            torch.nn.init.constant_(self.out_proj.bias, 0.0)


    def forward(self, hidden_states, past_key_value=None, use_cache=False, attention_mask=None):
        """
        Forward pass of the self-attention layer with Grouped Query Attention support.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].
            past_key_value (tuple, optional): Tuple of (past_key, past_value), each of shape
                [batch_size, num_key_value_heads, past_seq_len, head_dim]. Defaults to None.
            use_cache (bool, optional): Whether to return cached key/value for generation. Defaults to False.
            attention_mask (torch.Tensor, optional): Attention mask of shape
                [batch_size, seq_len] for padding. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, embed_dim].
            tuple: (output, past_key_value) if use_cache is True.
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Compute query, key, value projections
        query = self.query_proj(hidden_states)  # [batch_size, seq_len, embed_dim]
        key = self.key_proj(hidden_states)      # [batch_size, seq_len, head_dim * num_key_value_heads]
        value = self.value_proj(hidden_states)  # [batch_size, seq_len, head_dim * num_key_value_heads]

        # Reshape and transpose
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        if self.pos_encoding is not None:
            if past_key_value is not None:
                past_seq_len = past_key_value[0].size(2)
                input_pos = torch.arange(
                    past_seq_len,
                    past_seq_len + seq_len,
                    device=hidden_states.device
                )
            else:
                input_pos = None
            query = self.pos_encoding(query, position_ids=input_pos)
            key = self.pos_encoding(key, position_ids=input_pos)
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # For GQA: Repeat key and value to match num_heads if num_key_value_heads < num_heads
        if self.num_key_value_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_key_value_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        # Handle past key/value for incremental decoding
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)  # Concatenate along sequence dimension
            value = torch.cat([past_value, value], dim=2)
            is_causal = False  # No causal mask for past tokens (generation mode)
        else:
            is_causal = self.is_decoder  # Apply causal mask only if in decoder mode

        # Prepare attention mask
        attn_mask = None
        if is_causal:
            # Causal mask: -inf above diagonal (future), 0 on/below diagonal
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_len, seq_len]
        if attention_mask is not None:
            # Padding mask: -inf where attention_mask == 0, 0 where == 1
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            padding_mask = torch.where(padding_mask == 0, float('-inf'), 0.0)
        
        # Combine masks additively
        if is_causal and attention_mask is not None:
            attn_mask = causal_mask + padding_mask  # -inf where either mask has -inf
        elif is_causal:
            attn_mask = causal_mask
        elif attention_mask is not None:
            attn_mask = padding_mask
        
        if attn_mask is not None:
            attn_mask = attn_mask.to(query.dtype)
        
        # Compute attention using scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )

        # Reshape back: [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.out_proj(attn_output)

        if use_cache:
            return output, (key, value)
        return output


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, num_key_value_heads=None, bias=False, pos_encoding=None, num_layers=12):
        """
        Initialize the multi-headed cross-attention layer with support for Grouped Query Attention.

        Args:
            embed_dim (int): Embedding dimension of the input.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            num_key_value_heads (int, optional): Number of query groups for Grouped Query Attention.
                If None, defaults to num_heads (standard MHA). Defaults to None.
            bias (bool, optional): If True, adds bias to input/output projection layers. Defaults to False.
            pos_encoding (nn.Module, optional): Positional encoding module. Defaults to None.
            num_layers (int, optional): Number of layers in the model, used for weight initialization. Defaults to 12.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.dropout = dropout
        self.bias = bias
        self.num_layers = num_layers

        # Ensure num_heads is divisible by num_key_value_heads
        assert self.num_heads % self.num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_proj = nn.Linear(embed_dim, self.head_dim * self.num_key_value_heads, bias=bias)
        self.value_proj = nn.Linear(embed_dim, self.head_dim * self.num_key_value_heads, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = pos_encoding

        self._reset_parameters()

    def _reset_parameters(self):
        std = 0.02 / math.sqrt(self.num_layers)
        kv_std = std / (self.num_heads / self.num_key_value_heads) ** 0.5
        torch.nn.init.normal_(self.query_proj.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.key_proj.weight, mean=0.0, std=kv_std)
        torch.nn.init.normal_(self.value_proj.weight, mean=0.0, std=kv_std)
        torch.nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)

        if self.bias:
            torch.nn.init.constant_(self.query_proj.bias, 0.0)
            torch.nn.init.constant_(self.key_proj.bias, 0.0)
            torch.nn.init.constant_(self.value_proj.bias, 0.0)
            torch.nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, hidden_states, encoder_output, encoder_mask=None):
        """
        Forward pass of the cross-attention layer with Grouped Query Attention support.

        Args:
            hidden_states (torch.Tensor): Decoder input tensor of shape [batch_size, tgt_seq_len, embed_dim].
            encoder_output (torch.Tensor): Encoder output tensor of shape [batch_size, src_seq_len, embed_dim].
            encoder_mask (torch.Tensor, optional): Padding mask for encoder of shape [batch_size, src_seq_len].
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, tgt_seq_len, embed_dim].
        """
        batch_size, tgt_seq_len, _ = hidden_states.size()
        src_seq_len = encoder_output.size(1)

        # Compute projections
        query = self.query_proj(hidden_states)  # [batch_size, tgt_seq_len, embed_dim]
        key = self.key_proj(encoder_output)     # [batch_size, src_seq_len, head_dim * num_key_value_heads]
        value = self.value_proj(encoder_output) # [batch_size, src_seq_len, head_dim * num_key_value_heads]

        # Reshape before applying positional encoding
        query = query.view(batch_size, tgt_seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, src_seq_len, self.num_key_value_heads, self.head_dim)
        value = value.view(batch_size, src_seq_len, self.num_key_value_heads, self.head_dim)


        # Apply positional encoding to query and key
        if self.pos_encoding is not None:
            query = self.pos_encoding(query)  # [batch_size, tgt_seq_len, num_heads, head_dim]
            key = self.pos_encoding(key)      # [batch_size, src_seq_len, num_key_value_heads, head_dim]
        
        # Reshape for attention
        query = query.transpose(1, 2)  # [batch_size, num_heads, tgt_seq_len, head_dim]
        key = key.transpose(1, 2)      # [batch_size, num_key_value_heads, src_seq_len, head_dim]
        value = value.transpose(1, 2)  # [batch_size, num_key_value_heads, src_seq_len, head_dim]

        # For GQA: Repeat key and value to match num_heads if num_key_value_heads < num_heads
        if self.num_key_value_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_key_value_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)
        
        # Prepare attention mask
        attn_mask = None
        if encoder_mask is not None:
            # Padding mask: -inf where attention_mask == 0, 0 where == 1
            padding_mask = encoder_mask.unsqueeze(1).unsqueeze(2).to(query.dtype)
            padding_mask = torch.where(padding_mask == 0, float('-inf'), 0.0)
            attn_mask = padding_mask
        
        if attn_mask is not None:
            attn_mask = attn_mask.to(query.dtype)
        
        # Compute attention using scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        # Reshape back: [batch_size, num_heads, tgt_seq_len, head_dim] -> [batch_size, tgt_seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_seq_len, self.embed_dim)
        # Final output projection
        output = self.out_proj(attn_output)

        return output
