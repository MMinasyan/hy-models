import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from typing import Optional, Tuple, Union
import math
import copy
from .modeling_components import MultiLayerPerceptron, MultiHeadSelfAttention, RotaryPositionalEmbeddings
from .configuration import ArtConfig


class ArtDecoderLayer(nn.Module):
    """
    A single decoder layer for the Art model, consisting of self-attention, MLP, and normalization.

    Args:
        hidden_size (int): Dimension of the input and output.
        num_heads (int): Number of attention heads.
        intermediate_dim (int): Dimension of the intermediate MLP layer.
        num_key_value_heads (int, optional): Number of query groups for Grouped Query Attention. Defaults to None.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        bias (bool, optional): Whether to add bias to linear layers. Defaults to False.
        mlp (nn.Module, optional): Custom MLP module. If None, uses default MultiLayerPerceptron. Defaults to None.
        norm (nn.Module, optional): Custom normalization module. If None, uses RMSNorm. Defaults to None.
        layer_norm_eps (float, optional): Epsilon for layer normalization. Defaults to 1e-5.
        pos_encoding (nn.Module, optional): Positional encoding module. Defaults to None.
    """
    def __init__(self, config: ArtConfig,
                 mlp: nn.Module = None, norm: nn.Module = None, pos_encoding=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.intermediate_dim = config.intermediate_dim
        self.dropout = config.dropout
        self.bias = config.bias
        self.layer_norm_eps = config.layer_norm_eps
        self.num_key_value_heads = config.num_key_value_heads


        self.self_attn = MultiHeadSelfAttention(
            config,
            is_decoder=True,
            pos_encoding=pos_encoding  # For RoPE
        )

        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = MultiLayerPerceptron(
                hidden_size=config.hidden_size,
                intermediate_dim=config.intermediate_dim,
                bias=config.bias
            )

        if norm is not None:
            self.norm1 = copy.deepcopy(norm)
            self.norm2 = copy.deepcopy(norm)
        else:
            self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout layer
        self.dropout_layer = nn.Dropout(config.dropout)
    
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


class ArtModel(nn.Module):
    def __init__(self, config: ArtConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # RoPE
        self.pos_encoding = RotaryPositionalEmbeddings(
            dim=config.hidden_size//config.num_heads,
            max_position=config.max_position_embeddings,
            base=10000.
            )
        # Stack of ArtDecoderLayers
        self.layers = nn.ModuleList([
            ArtDecoderLayer(
                config=config,
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

        if attention_mask is None:
            attention_mask = hidden_states.sum(dim=-1) == 0
            attention_mask = attention_mask.to(hidden_states.dtype)

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
    """
    The Art model for causal language modeling, compatible with Hugging Face Transformers.

    This class wraps the ArtModel and adds a language modeling head for next-token prediction.

    Args:
        config (ArtConfig): Configuration object containing model hyperparameters.
    """
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
        """
        Forward pass of the Art model for causal language modeling.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs of shape [batch_size, seq_len]. Defaults to None.
            attention_mask (torch.Tensor, optional): Padding mask of shape [batch_size, seq_len]. Defaults to None.
            position_ids (torch.LongTensor, optional): Position indices for input tokens. Defaults to None.
            past_key_values (tuple, optional): Cached key-value states for each layer. Defaults to None.
            inputs_embeds (torch.FloatTensor, optional): Pre-computed input embeddings of shape [batch_size, seq_len, hidden_size]. Defaults to None.
            labels (torch.LongTensor, optional): Target token IDs for computing loss. Defaults to None.
            use_cache (bool, optional): If True, returns past key-value states. Defaults to None.
            output_attentions (bool, optional): If True, returns attention weights. Defaults to None.
            output_hidden_states (bool, optional): If True, returns all hidden states. Defaults to None.
            return_dict (bool, optional): If True, returns a CausalLMOutputWithPast object. Defaults to None.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]: Model outputs containing:
                - loss (torch.FloatTensor, optional): Loss if labels are provided.
                - logits (torch.FloatTensor): Prediction scores.
                - past_key_values (tuple, optional): Past key-value states if use_cache is True.
                - hidden_states (tuple, optional): All hidden states if output_hidden_states is True.
                - attentions (tuple, optional): All attention weights if output_attentions is True.
        """
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
            output = (logits,)
            if use_cache is not None:
                output += (past_key_values,)
            if output_hidden_states:
                output += (all_hidden_states,)
            if output_attentions:
                output += (attentions,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=attentions
        )
