import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutput, Seq2SeqLMOutput
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from typing import Optional, Tuple, Union
from torchtune.modules import RotaryPositionalEmbeddings
import math
import copy
from modeling_components import MultiLayerPerceptron, MultiHeadSelfAttention, MultiHeadCrossAttention, Conv1dEmbedding
from configuration import AutoEditConfig


def build_embedding(config):
    if config.embedding_type == 'conv':
        return Conv1dEmbedding(embed_dim=256, hidden_size=config.hidden_size, vocab_size=320, inter_dim=max(512, config.hidden_size//2), kernel_size=5, dropout=config.dropout)
    
    elif config.embedding_type == 'token':
        return nn.Embedding(config.encoder_vocab_size, config.hidden_size)
    
    else:
        raise ValueError(f"Invalid embedding_type: {config.empedding_type}. Expected 'token' or 'conv'.")


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
        # std = 1 / math.sqrt(self.hidden_size * self.num_layers)
        # out_std = 1 / math.sqrt(self.intermediate_dim * self.num_layers)
        std = 0.02 / math.sqrt(self.num_layers)
        nn.init.normal_(self.w.weight, mean=0.0, std=std)
        nn.init.normal_(self.v.weight, mean=0.0, std=std)
        nn.init.normal_(self.out.weight, mean=0.0, std=std)  # Scaled for residual connections
        if self.bias == True:
            nn.init.constant_(self.w.bias, 0)
            nn.init.constant_(self.v.bias, 0)
            nn.init.constant_(self.out.bias, 0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.silu(self.w(x)) * self.v(x)
        return self.out(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, num_groups=None, bias=False, pos_encoding=None, num_layers=12):
        """
        Initialize the multi-headed cross-attention layer with support for Grouped Query Attention.

        Args:
            embed_dim (int): Embedding dimension of the input.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            num_groups (int, optional): Number of query groups for Grouped Query Attention.
                If None, defaults to num_heads (standard MHA).
            bias (bool): If True, adds bias to input/output projection layers.
            pos_encoding (nn.Module): Positional encoding module,
                should expect tensor with shape [batch_size, seq_len, num_heads(or num_groups), head_dim]
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups if num_groups is not None else num_heads
        self.dropout = dropout
        self.bias = bias
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
        # std = 0.02
        # kv_std = std / (self.num_heads / self.num_groups) ** 0.5
        # out_std = std / (2 * self.num_layers)**0.5

        std = 0.02 / math.sqrt(self.num_layers)
        kv_std = std / (self.num_heads / self.num_groups) ** 0.5
        torch.nn.init.normal_(self.query_proj.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.key_proj.weight, mean=0.0, std=kv_std)
        torch.nn.init.normal_(self.value_proj.weight, mean=0.0, std=kv_std)
        torch.nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)  # Scaled for residual connections

        # Handle biases if they exist (bias=True)
        if self.bias:
            torch.nn.init.constant_(self.query_proj.bias, 0.0)
            torch.nn.init.constant_(self.key_proj.bias, 0.0)
            torch.nn.init.constant_(self.value_proj.bias, 0.0)
            torch.nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, hidden_states, encoder_output, encoder_mask=None):
        """
        Forward pass of the cross-attention layer with Grouped Query Attention support.

        Args:
            hidden_states (torch.Tensor): Decoder input tensor, shape [batch_size, tgt_seq_len, embed_dim].
            encoder_output (torch.Tensor): Encoder output tensor, shape [batch_size, src_seq_len, embed_dim].
            encoder_mask (torch.Tensor, optional): Padding mask for encoder, shape [batch_size, src_seq_len].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, tgt_seq_len, embed_dim].
        """
        batch_size, tgt_seq_len, _ = hidden_states.size()
        src_seq_len = encoder_output.size(1)

        # Compute projections
        query = self.query_proj(hidden_states)  # [batch_size, tgt_seq_len, embed_dim]
        key = self.key_proj(encoder_output)     # [batch_size, src_seq_len, head_dim * num_groups]
        value = self.value_proj(encoder_output) # [batch_size, src_seq_len, head_dim * num_groups]

        # Reshape before applying positional encoding
        query = query.view(batch_size, tgt_seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, src_seq_len, self.num_groups, self.head_dim)
        value = value.view(batch_size, src_seq_len, self.num_groups, self.head_dim)


        # Apply positional encoding to query and key
        if self.pos_encoding is not None:
            query = self.pos_encoding(query)  # [batch_size, tgt_seq_len, num_heads, head_dim]
            key = self.pos_encoding(key)      # [batch_size, src_seq_len, num_groups, head_dim]
        
        # Reshape for attention
        query = query.transpose(1, 2)  # [batch_size, num_heads, tgt_seq_len, head_dim]
        key = key.transpose(1, 2)      # [batch_size, num_groups, src_seq_len, head_dim]
        value = value.transpose(1, 2)  # [batch_size, num_groups, src_seq_len, head_dim]

        # For GQA: Repeat key and value to match num_heads if num_groups < num_heads
        if self.num_groups != self.num_heads:
            repeat_factor = self.num_heads // self.num_groups
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


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_dim: int, num_groups: int = None, dropout: float = 0.1, bias: bool = False,
                 mlp: nn.Module = None, norm: nn.Module = None, layer_norm_eps: float = 1e-5, pos_encoding=None, num_layers=12):
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
            is_decoder=False,
            pos_encoding=pos_encoding,  # For RoPE
            num_layers=num_layers
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
    
    def forward(self, hidden_states, attention_mask=None):            
        x = self.norm1(hidden_states)
        x = self.self_attn( x, attention_mask=attention_mask)

        hidden_states = hidden_states + self.dropout_layer(x)

        x = self.norm2(hidden_states)
        x_mlp = self.mlp(x)
        x_mlp = self.dropout_layer(x_mlp)
        hidden_states = hidden_states + x_mlp

        return hidden_states


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_dim: int, num_groups: int = None, dropout: float = 0.1, bias: bool = False,
                 mlp: nn.Module = None, norm: nn.Module = None, layer_norm_eps: float = 1e-5, pos_encoding=None, num_layers=12):
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
            pos_encoding=pos_encoding,  # For RoPE
            num_layers=num_layers
        )
        # Cross-attention module (non-causal, attends to encoder output)
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout,
            bias=bias,
            # pos_encoding=pos_encoding
            num_layers=num_layers
        )

        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = MultiLayerPerceptron(
                hidden_size=hidden_size,
                intermediate_dim=intermediate_dim,
                bias=bias
            )

        # Normalization layers (pre-norm for each sub-layer)
        if norm is not None:
            self.norm1 = copy.deepcopy(norm)
            self.norm2 = copy.deepcopy(norm)
            self.norm3 = copy.deepcopy(norm)
        else:
            self.norm1 = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
            self.norm2 = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
            self.norm3 = nn.RMSNorm(hidden_size, eps=layer_norm_eps)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, hidden_states, encoder_output, attention_mask=None, encoder_mask=None, past_key_value=None, use_cache=False):            
        # Self-attention
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

        # Cross-attention
        x = self.norm2(hidden_states)
        x = self.cross_attn(
            x,
            encoder_output,
            encoder_mask=encoder_mask
        )
        hidden_states = hidden_states + self.dropout_layer(x)

        # MLP
        x = self.norm3(hidden_states)
        x = self.mlp(x)
        hidden_states = hidden_states + self.dropout_layer(x)

        if use_cache:
            return hidden_states, past_key_value
        return hidden_states


class Encoder(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, shared_embedding: nn.Module = None, pos_encoding: nn.Module = None):
        """
        Initialize the Encoder module for a transformer model.

        Args:
            config (PretrainedConfig): Configuration object containing model hyperparameters.
            shared_embedding (nn.Module, optional): Shared embedding layer for token embeddings.
                If None, a new nn.Embedding layer is created. Defaults to None.
            pos_encoding (nn.Module, optional): Positional encoding module.
                If None, a default RotaryPositionalEmbeddings is initialized. Defaults to None.
        """
        super().__init__(config)
        # self.config = config
        # RoPE
        if pos_encoding is None:
            self.pos_encoding = RotaryPositionalEmbeddings(
                dim=config.hidden_size//config.num_heads,
                max_seq_len=config.max_position_embeddings,
                base=10000.
                )
        else:
            self.pos_encoding = pos_encoding

        # Token embeddings
        if shared_embedding is None:
            self.embed_tokens = build_embedding(config, self.pos_encoding)
        else:
            self.embed_tokens = shared_embedding
        
        # Stack of EncoderLayers
        num_layers = config.num_encoder_layers - 2 if config.embedding_type=='stage1' else config.num_encoder_layers

        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_dim=config.intermediate_dim,
                num_groups=config.num_groups,
                dropout=config.dropout,
                bias=config.bias,
                layer_norm_eps=config.layer_norm_eps,
                pos_encoding=self.pos_encoding,  # Pass pos_encoding for RoPE
                num_layers=config.num_encoder_layers
            ) for _ in range(num_layers)
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
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass of the Encoder module.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs of shape [batch_size, seq_len]. Defaults to None.
            attention_mask (torch.Tensor, optional): Padding mask of shape [batch_size, seq_len]. Defaults to None.
            inputs_embeds (torch.FloatTensor, optional): Pre-computed embedded inputs of shape
                [batch_size, seq_len, hidden_size]. Defaults to None.
            output_hidden_states (bool, optional): If True, returns hidden states from all layers. Defaults to None.
            output_attentions (bool, optional): If True, returns attention weights. Defaults to None.
            return_dict (bool, optional): If True, returns a BaseModelOutput object. Defaults to None.

        Returns:
            Union[Tuple, BaseModelOutput]: Model outputs containing:
                - last_hidden_state (torch.FloatTensor): Final hidden states.
                - hidden_states (tuple, optional): All hidden states if output_hidden_states is True.
                - attentions (tuple, optional): All attention weights if output_attentions is True.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided, or if neither is provided.
        """
        # Set defaults
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

        # Collect hidden states if requested
        all_hidden_states = () if output_hidden_states else None

        if self.config.embedding_type == 'conv':
            attention_mask = (hidden_states.sum(dim=-1) > 0).long()

        # Pass through each layer
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask
            )

        # Final norm
        hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return results
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            return outputs

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=None
        )


class Decoder(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, shared_embedding: nn.Module = None, pos_encoding: nn.Module = None):
        """
        Initialize the Decoder module for an encoder-decoder transformer model.

        Args:
            config (PretrainedConfig): Configuration object containing model hyperparameters.
            shared_embedding (nn.Module, optional): Shared embedding layer for token embeddings.
                If None, a new Embedding layer is created. Defaults to None.
            pos_encoding (nn.Module, optional): Positional encoding module.
                If None, a default RotaryPositionalEmbeddings is initialized. Defaults to None.
        """
        super().__init__(config)
        # self.config = config

        # Token embeddings
        if shared_embedding is None:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = shared_embedding
        
        # RoPE
        if pos_encoding is None:
            self.pos_encoding = RotaryPositionalEmbeddings(
                dim=config.hidden_size//config.num_heads,
                max_seq_len=config.max_position_embeddings,
                base=10000.
                )
        else:
            self.pos_encoding = pos_encoding
        
        # Stack of EncoderLayers
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_dim=config.intermediate_dim,
                num_groups=config.num_groups,
                dropout=config.dropout,
                bias=config.bias,
                layer_norm_eps=config.layer_norm_eps,
                pos_encoding=self.pos_encoding,  # Pass pos_encoding for RoPE
                num_layers=config.num_layers
            ) for _ in range(config.num_layers)
        ])

        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass of the Decoder module.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs of shape [batch_size, tgt_seq_len]. Defaults to None.
            attention_mask (torch.Tensor, optional): Padding mask of shape [batch_size, tgt_seq_len]. Defaults to None.
            encoder_hidden_states (torch.Tensor, optional): Encoder output of shape
                [batch_size, src_seq_len, hidden_size]. Defaults to None.
            encoder_attention_mask (torch.Tensor, optional): Padding mask for encoder of shape
                [batch_size, src_seq_len]. Defaults to None.
            past_key_values (tuple, optional): Past key-value pairs for each layer. Defaults to None.
            use_cache (bool, optional): If True, returns past key-value pairs. Defaults to False.
            inputs_embeds (torch.FloatTensor, optional): Pre-computed embedded inputs of shape
                [batch_size, tgt_seq_len, hidden_size]. Defaults to None.
            output_hidden_states (bool, optional): If True, returns all hidden states. Defaults to None.
            output_attentions (bool, optional): If True, returns attention weights. Defaults to None.
            return_dict (bool, optional): If True, returns a BaseModelOutputWithPast object. Defaults to None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: Model outputs containing:
                - last_hidden_state (torch.FloatTensor): Final hidden states.
                - past_key_values (tuple, optional): Past key-value pairs if use_cache is True.
                - hidden_states (tuple, optional): All hidden states if output_hidden_states is True.
                - attentions (tuple, optional): All attention weights if output_attentions is True.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided, or if neither is provided.
        """
        # Set defaults
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
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

        # Collect hidden states if requested
        all_hidden_states = () if output_hidden_states else None

        # Initialize new_past_key_values if caching is enabled
        if use_cache:
            new_past_key_values = []
        else:
            new_past_key_values = None

        # Pass through each layer
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Extract past key-value for this layer, if provided
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_output = layer(
                hidden_states=hidden_states,
                encoder_output=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_mask=encoder_attention_mask,
                past_key_value=layer_past_key_value,
                use_cache=use_cache
            )

            # Handle output based on whether caching is used
            if use_cache:
                hidden_states, layer_past_key_value = layer_output
                new_past_key_values.append(layer_past_key_value)
            else:
                hidden_states = layer_output

        # Final norm
        hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return results
        if not return_dict:
            if use_cache:
                return (hidden_states, tuple(new_past_key_values))
            else:
                outputs = (hidden_states,)
                if output_hidden_states:
                    outputs += (all_hidden_states,)
                return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(new_past_key_values) if use_cache else None,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=None
        )


class AutoEditForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """
    An encoder-decoder model compatible with the transformers library and GenerationMixin.
    Integrates Encoder and Decoder classes with a shared embedding and language modeling head.
    """
    
    def __init__(self, config: AutoEditConfig):
        """
        Initialize the AutoEdit model for conditional generation.

        Args:
            config (AutoEditConfig): Configuration object containing model hyperparameters.
        """
        super().__init__(config)

        if config.num_encoder_layers is None:
            config.num_encoder_layers = config.num_layers
        
        # Shared embedding layer for encoder, decoder, and LM head
        self.shared_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self._initialize_shared_embedding()
        
        self.pos_encoding = RotaryPositionalEmbeddings(
            dim=config.hidden_size//config.num_heads,
            max_seq_len=config.max_position_embeddings,
            base=10000.
        )

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        if self.config.embedding_type == 'token':
            self.encoder = Encoder(
                encoder_config,
                shared_embedding=self.shared_embedding if config.tie_encoder_weights else None,
                pos_encoding=self.pos_encoding if config.tie_encoder_weights else None
                )
        else:
            self.encoder = Encoder(encoder_config, pos_encoding=self.pos_encoding)
        

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        self.decoder = Decoder(decoder_config, self.shared_embedding, pos_encoding=self.pos_encoding)

        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        if config.tie_weights:
            self.tie_weights()
        else:
            self._initialize_lm_head()

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
    def tie_weights(self):
        self.lm_head.weight = self.shared_embedding.weight

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def _initialize_shared_embedding(self):
        std = math.sqrt(1 / self.config.hidden_size)
        nn.init.normal_(self.shared_embedding.weight, mean=0.0, std=std)
    
    def _initialize_lm_head(self):
        std = math.sqrt(1 / self.config.hidden_size)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[BaseModelOutput, Tuple]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = False,
        **kwargs
    ) -> Union[dict, Tuple]:
        """
        Forward pass of the AutoEdit model.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs of shape [batch_size, src_seq_len]. Defaults to None.
            attention_mask (torch.Tensor, optional): Padding mask of shape [batch_size, src_seq_len]. Defaults to None.
            decoder_input_ids (torch.LongTensor, optional): Decoder input token IDs of shape
                [batch_size, tgt_seq_len]. Defaults to None.
            decoder_attention_mask (torch.Tensor, optional): Padding mask for decoder of shape
                [batch_size, tgt_seq_len]. Defaults to None.
            encoder_outputs (Union[BaseModelOutput, Tuple], optional): Pre-computed encoder outputs. Defaults to None.
            past_key_values (tuple, optional): Past key-value pairs for each layer. Defaults to None.
            use_cache (bool, optional): If True, returns past key-value pairs. Defaults to False.
            output_attentions (bool, optional): If True, returns attention weights. Defaults to False.
            labels (torch.LongTensor, optional): Target token IDs of shape [batch_size, tgt_seq_len]. Defaults to None.
            return_dict (bool, optional): If True, returns a dictionary. Defaults to False.
            **kwargs: Additional arguments for compatibility.

        Returns:
            Union[dict, Tuple]: Model outputs containing:
                - loss (torch.FloatTensor, optional): Loss if labels are provided.
                - logits (torch.FloatTensor): Prediction scores.
                - past_key_values (tuple, optional): Past key-value pairs if use_cache is True.
                - decoder_hidden_states (tuple, optional): All decoder hidden states.
                - encoder_hidden_states (torch.FloatTensor): Final encoder hidden states.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided, or if neither is provided.
        """
        # Compute encoder outputs if not provided
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
        else:
            # Handle both BaseModelOutput and tuple formats
            encoder_hidden_states = (
                encoder_outputs.last_hidden_state
                if isinstance(encoder_outputs, BaseModelOutput)
                else encoder_outputs[0]
            )

        # Decoder forward pass with encoder outputs
        if self.config.embedding_type == 'conv':
            attention_mask = (encoder_hidden_states.sum(dim=-1) > 0).long()

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True
        )

        # Extract decoder hidden states
        decoder_hidden_states = decoder_outputs.last_hidden_state

        # Compute logits using the language modeling head
        logits = self.lm_head(decoder_hidden_states)

        # Compute loss if labels are provided (e.g., during training)
        loss = None
        if labels is not None:
            # Shift logits and labels for teacher forcing
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Cross-entropy loss, ignoring padding tokens (assumed as -100)
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Prepare outputs based on return_dict
        if return_dict:
            return Seq2SeqLMOutput(
                loss=loss,
                logits=logits,
                past_key_values=decoder_outputs.past_key_values if use_cache else None,
                decoder_hidden_states=decoder_hidden_states,
                decoder_attentions=None, # To do
                cross_attentions=None, # To do
                encoder_last_hidden_state=encoder_hidden_states,
            )
        else:
            outputs = (logits,)
            if use_cache:
                outputs += (decoder_outputs.past_key_values,)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
    
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # Each layer_past is a tuple of (key, value) tensors
            reordered_layer = tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            reordered_past += (reordered_layer,)
        return reordered_past

