import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union, Callable, Optional
from transformers import PretrainedConfig, PreTrainedModel
from torchtune.modules import RotaryPositionalEmbeddings


class ConvEmbedding(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim, num_filters, out_dim, dropout=0.1, padding_idx=0):
        super().__init__()
        self.char_embed_dim = char_embed_dim
        self.embedding = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=self.char_embed_dim, padding_idx=padding_idx)
        self.conv1 = nn.Conv1d(in_channels=self.char_embed_dim, out_channels=num_filters, kernel_size=3, padding='same')
        self.gelu1 = nn.GELU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=out_dim, kernel_size=3, padding='same')
        self.gelu2 = nn.GELU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.padding_idx = padding_idx

    def forward(self, x):
        # Assume x has shape [batch_size, num_words, word_length]
        batch_size, num_words, word_length = x.shape
        mask = (x.sum(-1) > 0).float().unsqueeze(-1) # Shape: [batch_size, num_words, 1]

        x = self.embedding(x) # [batch_size, num_words, word_length, char_embed_dim]
        x = x.reshape(-1, word_length, self.char_embed_dim) # [batch_size * num_words, word_length, char_embed_dim]
        x = x.permute(0, 2, 1) # [batch_size * num_words, char_embed_dim, word_length]

        x = self.conv1(x) # [batch_size * num_words, num_filters, word_length]
        x = self.gelu1(x)

        x = self.pool1(x) # [batch_size * num_words, num_filters, word_length // 2]
        x = self.dropout(x)

        x = self.conv2(x) # [batch_size * num_words, out_dim, word_length // 2]
        x = self.gelu2(x)

        x = self.max_pool(x) # [batch_size * num_words, out_dim, 1]
        x = x.squeeze(-1) # [batch_size * num_words, out_dim]
        x = x.reshape(batch_size, num_words, -1) # [batch_size, num_words, out_dim]

        return x * mask


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 self_attn: Callable[[], nn.Module] = None, norm: Callable[[], nn.Module] = None,
                 linear1: Callable[[], nn.Module] = None, linear2: Callable[[], nn.Module] = None,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # super().__init__()
        nn.Module.__init__(self)
        if self_attn is None:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                bias=bias, batch_first=batch_first,
                                                **factory_kwargs)
        else:
            self.self_attn = self_attn()
        
        # Implementation of Feedforward model
        if linear1 is None:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        else:
            self.linear1 = linear1()
        if linear2 is None:
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        else:
            self.linear2 = linear2()
        self.dropout = nn.Dropout(dropout)

        self.norm_first = norm_first
        if norm is None:
            self.norm1 = nn.RMSNorm(d_model, eps=layer_norm_eps, elementwise_affine=True)
            self.norm2 = nn.RMSNorm(d_model, eps=layer_norm_eps, elementwise_affine=True)
        else:
            self.norm1 = norm()
            self.norm2 = norm()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            if activation == "relu":
                self.activation =  F.relu
            elif activation == "gelu":
                self.activation =  F.gelu
        else:
            self.activation = activation
        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu
    
    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x



class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 self_attn: Callable[[], nn.Module] = None, multihead_attn: Callable[[], nn.Module] = None,
                 norm: Callable[[], nn.Module] = None, linear1: Callable[[], nn.Module] = None,
                 linear2: Callable[[], nn.Module] = None, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        nn.Module.__init__(self)

        # Self-Attention
        if self_attn is None:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                   bias=bias, **factory_kwargs)
        else:
            self.self_attn = self_attn()
        # Cross-Attention
        if multihead_attn is None:
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                        bias=bias, **factory_kwargs)
        else:
            self.multihead_attn = multihead_attn()
        
        # Implementation of Feedforward model
        if linear1 is None:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        else:
            self.linear1 = linear1()
        if linear2 is None:
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        else:
            self.linear2 = linear2()
        self.dropout = nn.Dropout(dropout)

        self.norm_first = norm_first
        if norm is None:
            self.norm1 = nn.RMSNorm(d_model, eps=layer_norm_eps, elementwise_affine=True)
            self.norm2 = nn.RMSNorm(d_model, eps=layer_norm_eps, elementwise_affine=True)
            self.norm3 = nn.RMSNorm(d_model, eps=layer_norm_eps, elementwise_affine=True)
        else:
            self.norm1 = norm()
            self.norm2 = norm()
            self.norm3 = norm()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            if activation == "relu":
                self.activation =  F.relu
            elif activation == "gelu":
                self.activation =  F.gelu
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)


class SwiGLUFF(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.w = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.v = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(self.w(x)) * self.v(x)


# class RMSNorm(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.eps = eps
#         self.scale = nn.Parameter(torch.ones(dim))

#     def forward(self, x: torch.Tensor) ->torch. Tensor:
#         x_normed = (
#             x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#         )
#         return x_normed * self.scale


# class MHAwRoPE(nn.MultiheadAttention):
#     def __init__(self, embed_dim, num_heads, rope, rope_decoder=None, **kwargs):
#         nn.Module.__init__(self)
#         self.mha = nn.MultiheadAttention(embed_dim, num_heads, **kwargs)
#         self.batch_first = self.mha.batch_first
#         self._qkv_same_embed_dim = self.mha._qkv_same_embed_dim
#         self.embed_dim = self.mha.embed_dim
#         self.num_heads = self.mha.num_heads
#         self.in_proj_bias = self.mha.in_proj_bias
#         self.rope = rope
#         self.rope_decoder = rope_decoder

#     def forward(self, query, key, value, **kwargs):
#         # print(f'embed_dim: {self.embed_dim}, num_heads: {self.num_heads}')
#         q_shape, k_shape = query.size(), key.size()
#         query = query.view(query.size(0), query.size(1), self.num_heads, self.rope.dim)
#         key = key.view(key.size(0), key.size(1), self.num_heads, self.rope.dim)
#         if self.rope_decoder is None:
#             # print(f'query shape: {query.shape}')
#             query = self.rope(query)  # Apply RoPE to query
#         else:
#             query = self.rope_decoder(query)
#         key = self.rope(key)
#         return self.mha(
#             query.view(q_shape),
#             key.view(k_shape),
#             value,
#             **kwargs
#             )


class MHAwRoPE(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, rope, rope_decoder=None, bias=False, **kwargs):
        super().__init__(embed_dim, num_heads, bias=bias, **kwargs)
        self.rope = rope
        self.rope_decoder = rope_decoder

    def forward(self, query, key, value, **kwargs):
        q_shape, k_shape = query.size(), key.size()
        query = query.view(query.size(0), query.size(1), self.num_heads, self.rope.dim)
        key = key.view(key.size(0), key.size(1), self.num_heads, self.rope.dim)
        if self.rope_decoder is None:
            query = self.rope(query)  # Apply RoPE to query
        else:
            query = self.rope_decoder(query)   # Apply RoPE to keys and values
        key = self.rope(key)

        return super().forward(
            query.view(q_shape),
            key.view(k_shape),
            value,
            **kwargs
            )



class HyCorrConfig(PretrainedConfig):
    model_type = "hycorr"

    def __init__(
        self, 
        hidden_dim=768,
        num_heads=12, 
        num_layers=12,
        char_vocab_size=512,
        vocab_size=50000, 
        dropout=0.1, 
        max_length=512,
        use_swiglu=False,
        intermediate_dim=None,
        bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.char_vocab_size = char_vocab_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_length = max_length
        self.use_swiglu = use_swiglu
        if intermediate_dim is None:
            self.intermediate_dim = self.hidden_dim * 4
        else:
            self.intermediate_dim = intermediate_dim
        self.bias = bias


class HyCorr(PreTrainedModel):
    config_class = HyCorrConfig

    def __init__(self, config):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=0)
        self.rope = RotaryPositionalEmbeddings(dim=config.hidden_dim//config.num_heads, max_seq_len=config.max_length, base=10000.)

        def _build_sa_encoder():
            return MHAwRoPE(config.hidden_dim, config.num_heads, rope=self.rope, dropout=config.dropout, bias=config.bias, batch_first=True)
        def _build_sa_decoder():
            return MHAwRoPE(config.hidden_dim, config.num_heads, rope=self.rope, dropout=config.dropout, bias=config.bias, batch_first=True)
        def _build_ca_decoder():
            return MHAwRoPE(config.hidden_dim, config.num_heads, rope=self.rope, rope_decoder=self.rope, dropout=config.dropout, bias=config.bias, batch_first=True)
        def _build_linear1():
            return SwiGLUFF(hidden_dim=config.hidden_dim, intermediate_dim=config.intermediate_dim, bias=config.bias)
        # def _build_linear2():
        #     return nn.Identity()
        # def _build_norm():
        #     return RMSNorm(hidden_dim)
        # Encoder components
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            self_attn=_build_sa_encoder,
            linear1=_build_linear1 if self.config.use_swiglu else None,
            # linear2=_build_linear2,
            # norm=_build_norm,
            dim_feedforward=config.intermediate_dim,
            dropout=config.dropout,
            activation=nn.Identity() if self.config.use_swiglu else nn.GELU(),
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=True,
            bias=config.bias
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        decoder_layer = TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            self_attn=_build_sa_decoder,
            multihead_attn=_build_ca_decoder,
            linear1=_build_linear1 if self.config.use_swiglu else None,
            # linear2=_build_linear2,
            # norm=_build_norm,
            dim_feedforward=config.intermediate_dim,
            dropout=config.dropout,
            activation=nn.Identity() if self.config.use_swiglu else nn.GELU(),
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=True,
            bias=config.bias
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)

        self.output_layer = torch.nn.Linear(config.hidden_dim, config.vocab_size)
        
        causal_mask = torch.triu(torch.ones((config.max_length, config.max_length), dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)

        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)


    def forward(self, input_ids, decoder_input_ids, attention_mask=None, decoder_attention_mask=None, labels=None):
        if attention_mask is not None:
            attention_mask = attention_mask == 0
        
        encoder_embeddings = self.embedding(input_ids)

        encoder_outputs = self.encoder(encoder_embeddings, src_key_padding_mask=attention_mask)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask == 0

        decoder_embeddings = self.embedding(decoder_input_ids)
        # decoder_embeddings = decoder_embeddings + position_embeddings[:,:decoder_input_ids.size(1)]

        decoder_outputs = self.decoder(
            decoder_embeddings,
            memory=encoder_outputs,
            tgt_mask=self.causal_mask[:decoder_input_ids.size(1),:decoder_input_ids.size(1)],
            memory_key_padding_mask=attention_mask,
            tgt_key_padding_mask=decoder_attention_mask
        )

        output_logits = self.output_layer(decoder_outputs)
        output_logits = output_logits.permute(0, 2, 1)
        
        outputs = {'logits': output_logits}
        if labels is not None:
            loss = self.loss_fct(output_logits, labels)
            outputs['loss'] = loss
        
        return outputs


