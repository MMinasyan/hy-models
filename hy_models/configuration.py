from transformers import PretrainedConfig, CONFIG_MAPPING


class ArtConfig(PretrainedConfig):
    """
    Configuration class for the autoregressive transformer model.
    
    This class contains all the hyperparameters required to initialize an autoregressive transformer model.
    """
    model_type = "art"
    def __init__(
        self,
        hidden_size=768,
        num_heads=12, 
        num_layers=12,
        num_key_value_heads=None,
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
        """
        Initialize the autoregressive transformer model configuration.

        Args:
            hidden_size (int, optional): Dimensionality of the hidden states. Defaults to 768.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            num_layers (int, optional): Number of hidden layers in the model. Defaults to 12.
            num_key_value_heads (int, optional): Number of key/value heads for grouped attention.
                If None, equal to num_heads (no grouped attention). Defaults to None.
            vocab_size (int, optional): Vocabulary size. Defaults to 50000.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            intermediate_dim (int, optional): Dimensionality of the feed-forward intermediate layer. Defaults to 3584.
            bias (bool, optional): Whether to use bias terms in linear layers. Defaults to False.
            max_position_embeddings (int, optional): Maximum sequence length supported by model. Defaults to 2048.
            layer_norm_eps (float, optional): Epsilon for layer normalization. Defaults to 1e-5.
            tie_weights (bool, optional): Whether to tie input and output embeddings. Defaults to True.
            bos_token_id (int, optional): Beginning of sequence token ID. Defaults to 2.
            pad_token_id (int, optional): Padding token ID. Defaults to 0.
            eos_token_id (int, optional): End of sequence token ID. Defaults to 3.
            **kwargs: Additional parameters to be passed to the parent class.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.num_key_value_heads = num_key_value_heads
        self.dropout = dropout
        self.bias = bias
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.tie_weights = tie_weights
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self._attn_implementation = kwargs.get("attn_implementation", "sdpa")


CONFIG_MAPPING.register("art", ArtConfig)


class AutoEditConfig(PretrainedConfig):
    """
    Configuration class for the AutoEdit model.
    
    This class contains all the hyperparameters required to initialize an encoder-decoder
    AutoEdit model for conditional generation tasks.
    """
    model_type = "autoedit"

    def __init__(
        self,
        hidden_size=768,
        num_heads=12, 
        num_layers=12,
        num_encoder_layers=None,
        num_key_value_heads=None,
        vocab_size=50000, 
        dropout=0.1,
        intermediate_dim=3072,
        bias=False,
        max_position_embeddings=1024,
        layer_norm_eps=1e-5,
        embedding_type='token',
        tie_weights=True,
        tie_encoder_weights=True,
        encoder_vocab_size=None,
        pad_token_id=0,
        eos_token_id=3,
        decoder_start_token_id=2,
        is_decoder=False,
        is_encoder_decoder=False,
        **kwargs
    ):
        """
        Initialize the AutoEdit model configuration.

        Args:
            hidden_size (int, optional): Dimensionality of the hidden states. Defaults to 768.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            num_layers (int, optional): Number of hidden layers in the decoder. Defaults to 12.
            num_encoder_layers (int, optional): Number of layers in the encoder.
                If None, uses num_layers. Defaults to None.
            num_key_value_heads (int, optional): Number of key/value heads for grouped attention.
                If None, equal to num_heads (no grouped attention). Defaults to None.
            vocab_size (int, optional): Vocabulary size. Defaults to 50000.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            intermediate_dim (int, optional): Dimensionality of the feed-forward intermediate layer. Defaults to 3072.
            bias (bool, optional): Whether to use bias terms in linear layers. Defaults to False.
            max_position_embeddings (int, optional): Maximum sequence length supported by model. Defaults to 1024.
            layer_norm_eps (float, optional): Epsilon for layer normalization. Defaults to 1e-5.
            embedding_type (str, optional): Type of embedding to use ('token' or 'conv'). Defaults to 'token'.
            tie_weights (bool, optional): Whether to tie decoder input and output embeddings. Defaults to True.
            tie_encoder_weights (bool, optional): Whether to tie encoder and decoder embeddings. Defaults to True.
            encoder_vocab_size (int, optional): Encoder vocabulary size. If None, uses vocab_size. Defaults to None.
            pad_token_id (int, optional): Padding token ID. Defaults to 0.
            eos_token_id (int, optional): End of sequence token ID. Defaults to 3.
            decoder_start_token_id (int, optional): Initial token ID for decoder. Defaults to 2.
            is_decoder (bool, optional): Whether the model is used as a decoder. Defaults to False.
            is_encoder_decoder (bool, optional): Whether the model is used as an encoder-decoder. Defaults to False.
            **kwargs: Additional parameters to be passed to the parent class.
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.intermediate_dim = intermediate_dim
        self.bias = bias
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.embedding_type = embedding_type
        self.tie_weights = tie_weights
        self.tie_encoder_weights = tie_encoder_weights
        self.encoder_vocab_size = encoder_vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.is_decoder = is_decoder
        self.is_encoder_decoder = is_encoder_decoder

CONFIG_MAPPING.register("autoedit", AutoEditConfig)
