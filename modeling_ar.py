import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from typing import Optional, Tuple, Union
from torchtune.modules import RotaryPositionalEmbeddings
import math
import copy


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
        # Initialize w and v with standard deviation = 0.02
        nn.init.normal_(self.w.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.02 / (2 * self.num_layers)**0.5)  # Scaled for residual connections
        if self.bias == True:
            nn.init.constant_(self.w.bias, 0)
            nn.init.constant_(self.v.bias, 0)
            nn.init.constant_(self.out.bias, 0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.silu(self.w(x)) * self.v(x)
        return self.out(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, num_groups=None, bias=False, is_decoder=False, pos_encoding=None, num_layers=12):
        """
        Initialize the multi-headed self-attention layer with support for Grouped Query Attention.

        Args:
            embed_dim (int): Embedding dimension of the input.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            num_groups (int, optional): Number of query groups for Grouped Query Attention.
                                        If None, defaults to num_heads (standard MHA).
            bias (bool): If True, adds bias to input/output projection layers.
            is_decoder (bool): If True, applies causal attention (decoder mode).
                               If False, applies full attention (encoder mode).
            pos_encoding (nn.Module): positional encoding
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups if num_groups is not None else num_heads
        self.dropout = dropout
        self.bias = bias
        self.is_decoder = is_decoder  # New parameter to control causal attention
        self.num_layers = num_layers

        # Ensure num_heads is divisible by num_groups
        assert self.num_heads % self.num_groups == 0, "num_heads must be divisible by num_groups"
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_proj = nn.Linear(embed_dim, self.head_dim * self.num_groups, bias=bias)
        self.value_proj = nn.Linear(embed_dim, self.head_dim * self.num_groups, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = pos_encoding

        self._reset_parameters()

    def _reset_parameters(self):
        std = 0.02
        kv_std = std / (self.num_heads / self.num_groups) ** 0.5
        out_std = std / (2 * self.num_layers)**0.5
        torch.nn.init.normal_(self.query_proj.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.key_proj.weight, mean=0.0, std=kv_std)
        torch.nn.init.normal_(self.value_proj.weight, mean=0.0, std=kv_std)
        torch.nn.init.normal_(self.out_proj.weight, mean=0.0, std=out_std)  # Scaled for residual connections

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
                [batch_size, num_groups, past_seq_len, head_dim].
            use_cache (bool): Whether to return cached key/value for generation.
            attention_mask (torch.Tensor, optional): Attention mask of shape
                [batch_size, seq_len] for padding.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, embed_dim].
            tuple: (output, past_key_value) if use_cache is True.
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Compute query, key, value projections
        query = self.query_proj(hidden_states)  # [batch_size, seq_len, embed_dim]
        key = self.key_proj(hidden_states)      # [batch_size, seq_len, head_dim * num_groups]
        value = self.value_proj(hidden_states)  # [batch_size, seq_len, head_dim * num_groups]

        # Reshape and transpose
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_groups, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_groups, self.head_dim)

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
            query = self.pos_encoding(query, input_pos=input_pos)
            key = self.pos_encoding(key, input_pos=input_pos)

        # if self.pos_encoding is not None:
        #     query = self.pos_encoding(query)
        #     key = self.pos_encoding(key)
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # For GQA: Repeat key and value to match num_heads if num_groups < num_heads
        if self.num_groups != self.num_heads:
            repeat_factor = self.num_heads // self.num_groups
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
        # Final output projection
        output = self.out_proj(attn_output)

        # Handle caching
        # if use_cache:
        #     if self.num_groups != self.num_heads:
        #         key = key[:, ::repeat_factor, :, :]  # Take one per group
        #         value = value[:, ::repeat_factor, :, :]
        #     past_key_value = (key, value)
        #     return output, past_key_value
        if use_cache:
            return output, (key, value)  # Cache tensors with num_heads
        return output


class ArtDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_dim: int, num_groups: int = None, dropout: float = 0.1, bias: bool = False,
                 mlp: nn.Module = None, norm: nn.Module = None, layer_norm_eps: float = 1e-5, pos_encoding=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.bias = bias
        self.layer_norm_eps = layer_norm_eps
        self.num_groups = num_groups


        self.self_attn = MultiHeadSelfAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout,
            bias=bias,
            is_decoder=True,
            pos_encoding=pos_encoding  # For RoPE
        )

        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = MultiLayerPerceptron(
                hidden_size=hidden_size,
                intermediate_dim=intermediate_dim,
                bias=bias
            )

        if norm is not None:
            self.norm1 = copy.deepcopy(norm)
            self.norm2 = copy.deepcopy(norm)
        else:
            self.norm1 = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
            self.norm2 = nn.RMSNorm(hidden_size, eps=layer_norm_eps)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, hidden_states, past_key_value=None, use_cache=False, attention_mask=None):            
        x = self.norm1(hidden_states)
        x = self.self_attn(
            x,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask
        )

        if use_cache:
            x, past_key_value = x
        
        hidden_states = hidden_states + self.dropout_layer(x)

        x = self.norm2(hidden_states)
        x_mlp = self.mlp(x)
        x_mlp = self.dropout_layer(x_mlp)
        x = x + x_mlp

        if use_cache:
            return x, past_key_value
        return x


class ArtConfig(PretrainedConfig):
    model_type = "art"
    def __init__(
        self,
        hidden_size=768,
        num_heads=12, 
        num_layers=12,
        num_groups=None,
        vocab_size=50000,
        dropout=0.1,
        intermediate_dim=3584,
        bias=False,
        max_position_embeddings=2048,
        layer_norm_eps=1e-5,
        tie_weights=True,
        bos_token_id=2,
        pad_token_id=0,
        eos_token_id=3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.num_groups = num_groups
        self.dropout = dropout
        self.bias = bias
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.tie_weights = tie_weights
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


class ArtModel(nn.Module):
    def __init__(self, config: ArtConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # RoPE
        self.pos_encoding = RotaryPositionalEmbeddings(
            dim=config.hidden_size//config.num_heads,
            max_seq_len=config.max_position_embeddings,
            base=10000.
            )
        # Stack of ArtDecoderLayers
        self.layers = nn.ModuleList([
            ArtDecoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_dim=config.intermediate_dim,
                num_groups=config.num_groups,
                dropout=config.dropout,
                bias=config.bias,
                layer_norm_eps=config.layer_norm_eps,
                pos_encoding=self.pos_encoding  # Pass pos_encoding for RoPE
            ) for _ in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # Set defaults
        use_cache = use_cache if use_cache is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True

        # Handle input
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both")
        elif input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("You need to provide input_ids or inputs_embeds")

        # Prep past_key_values
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        # Collect hidden states if requested
        all_hidden_states = () if output_hidden_states else None
        next_cache = [] if use_cache else None

        # Pass through each layer
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_output = layer(
                hidden_states=hidden_states,
                past_key_value=past_key_values[i],
                use_cache=use_cache,
                attention_mask=attention_mask
            )

            # Handle layer output based on use_cache
            if use_cache:
                hidden_states, layer_cache = layer_output
                next_cache.append(layer_cache)
            else:
                hidden_states = layer_output

        # Final norm
        hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return results
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if use_cache:
                outputs += (tuple(next_cache),)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(next_cache) if use_cache else None,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=None
        )


class ArtForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ArtConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.transformer = ArtModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._initialize_embedding()

        if config.tie_weights:
            self.lm_head.weight = self.transformer.embed_tokens.weight

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    # Essential methods for Hugging Face compatibility
    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.transformer.embed_tokens = new_embeddings
        if self.config.tie_weights:
            self.lm_head.weight = new_embeddings.weight

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def get_decoder(self) -> nn.Module:
        return self.transformer

    def set_decoder(self, decoder: nn.Module):
        self.transformer = decoder

    def _initialize_embedding(self):
        std = math.sqrt(1 / self.transformer.embed_tokens.embedding_dim)
        nn.init.normal_(self.transformer.embed_tokens.weight, mean=0.0, std=std)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            # Only use the last token for incremental decoding
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True
        }
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        hidden_states = transformer_outputs.last_hidden_state
        if self.config.tie_weights:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            hidden_states = hidden_states * (self.config.hidden_size**-0.5)

        logits = self.lm_head(hidden_states)
        
        past_key_values = transformer_outputs.past_key_values if use_cache else None
        all_hidden_states = transformer_outputs.hidden_states if output_hidden_states else None
        attentions = None  # No attention weights available

        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + (past_key_values, all_hidden_states, attentions)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=attentions
        )
