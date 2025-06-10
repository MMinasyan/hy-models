from transformers import PretrainedConfig


class ArtConfig(PretrainedConfig):
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


class AutoEditConfig(PretrainedConfig):
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