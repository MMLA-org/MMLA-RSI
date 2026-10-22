"""Small random CPU model. No download, pretrained weights or benchmark."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from mmla_rsi.adapter import AdapterProfile, CausalLarcAdapter


class NeuralTests:
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        mixer = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
            intermediate_size=48, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
            attention_dropout=0.0))
        self.m = CausalLarcAdapter(mixer, AdapterProfile(32, 256, 8),
                                  seed=1, substrate_identity={"fixture": True})
