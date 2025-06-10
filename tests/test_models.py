import unittest
import torch
from torch import nn
from hy_models import ArtConfig, ArtForCausalLM, AutoEditConfig, AutoEditForConditionalGeneration
from hy_models.modeling_components import Conv1dEmbedding

class TestArtForCausalLM(unittest.TestCase):
    def setUp(self):
        self.config = ArtConfig(hidden_size=64, num_layers=2, num_heads=2, vocab_size=50)
        self.model = ArtForCausalLM(self.config)

    def test_initialization(self):
        self.assertEqual(self.model.config.hidden_size, 64)
        self.assertEqual(len(self.model.transformer.layers), 2)

    def test_forward(self):
        input_ids = torch.randint(0, 50, (1, 5))
        outputs = self.model(input_ids=input_ids, return_dict=True)
        self.assertEqual(outputs.logits.shape, (1, 5, 50))

    def test_forward_with_labels(self):
        input_ids = torch.randint(0, 50, (1, 5))
        labels = input_ids.clone()
        outputs = self.model(input_ids=input_ids, labels=labels, return_dict=True)
        self.assertIsNotNone(outputs.loss)

class TestAutoEditForConditionalGeneration(unittest.TestCase):
    def setUp(self):
        self.config = AutoEditConfig(hidden_size=64, num_layers=2, num_heads=2, vocab_size=50, embedding_type='token')
        self.model = AutoEditForConditionalGeneration(self.config)

    def test_initialization(self):
        self.assertEqual(self.model.config.hidden_size, 64)
        self.assertEqual(len(self.model.encoder.layers), 2)
        self.assertEqual(len(self.model.decoder.layers), 2)
        self.assertIsInstance(self.model.encoder.embed_tokens, nn.Embedding)

    def test_forward(self):
        input_ids = torch.randint(0, 50, (1, 5))
        decoder_input_ids = torch.randint(0, 50, (1, 3))
        outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
        self.assertEqual(outputs.logits.shape, (1, 3, 50))

    def test_forward_with_labels(self):
        input_ids = torch.randint(0, 50, (1, 5))
        decoder_input_ids = torch.randint(0, 50, (1, 3))
        labels = decoder_input_ids.clone()
        outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels, return_dict=True)
        self.assertIsNotNone(outputs.loss)

    def test_invalid_embedding_type(self):
        with self.assertRaises(ValueError):
            AutoEditForConditionalGeneration(AutoEditConfig(embedding_type='invalid'))

if __name__ == '__main__':
    unittest.main()
