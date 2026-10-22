import dataclasses, unittest
import torch
from mmla_rsi.adapter import AdapterProfile, CausalLarcAdapter
from mmla_rsi.factory import ProposalFactory, real_token_codec
from mmla_rsi.rules import Language, parse_rule
from mmla_rsi.evidence import EvidenceMemory, MemoryBounds
from mmla_rsi.updater import UpdateBounds
from tiny_helper import NeuralTests


class CodecTokenizer:
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    def encode(self, text, add_special_tokens=False):
        return [self.alphabet.index(c)+1 if c in self.alphabet else 53 for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(self.alphabet[i-1] if 1 <= i <= 52 else '?' for i in ids)


class AdapterTests(unittest.TestCase):
    def model(self):
        fixture=NeuralTests();fixture.setUp();return fixture.m
    def test_real_input_gradient_reset_and_frozen_backbone(self):
        model=self.model();fast=model.rho.begin(9)
        ids=torch.tensor([[1,3,4,5]])
        logits=model.forward_batch(ids,torch.ones_like(ids),fast_states=[fast])
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,64),ids[:,1:].reshape(-1))
        ga,gb=torch.autograd.grad(loss,(fast.a,fast.b))
        self.assertEqual(int(ga.count_nonzero()),0);self.assertGreater(float(gb.norm()),0)
        self.assertTrue(all(not x.requires_grad for x in model.mixer.parameters()))
        with torch.no_grad():model.rho.b.fill_(.02)
        reset=model.rho.reset(model.rho.begin(9));torch.testing.assert_close(reset.b,model.rho.b)
    def test_selected_logits_and_padding_match_full(self):
        model=self.model();ids=torch.tensor([[0,1,3,4]]);mask=torch.tensor([[0,1,1,1]])
        state=model.rho.begin(3);positions=torch.tensor([1,3])
        full=model.forward_batch(ids,mask,fast_states=[state])
        selected=model.forward_batch(ids,mask,fast_states=[state],logit_positions=positions)
        torch.testing.assert_close(selected,full[:,positions,:])
        unpadded=model.forward_batch(ids[:,1:],mask[:,1:],fast_states=[state])
        torch.testing.assert_close(full[:,1:,:],unpadded,atol=1e-6,rtol=1e-5)
    def test_invalid_batch_memory_positions_and_profile_rejected(self):
        model=self.model();ids=torch.ones((1,2),dtype=torch.long)
        with self.assertRaises(ValueError):model.forward_batch(ids.repeat(2,1),ids.repeat(2,1))
        with self.assertRaises(ValueError):model.forward_batch(ids,ids,memory={})
        with self.assertRaises(ValueError):model.forward_batch(ids,ids,logit_positions=torch.tensor([-1]))
        with self.assertRaises(ValueError):AdapterProfile(32,2,3)
    def test_factory_all_modes_isolate_state_and_reject_changed_reader(self):
        model=self.model();model.profile=dataclasses.replace(model.profile,local_tokens=8192)
        tokenizer=CodecTokenizer();language=Language(max_nodes=12,max_depth=1,max_tokens=6)
        memory=EvidenceMemory(MemoryBounds(slots=1,read_tokens=1100),reader_identity='codec-v1',count_tokens=len,namespace='proposal')
        factory=ProposalFactory(model,tokenizer,tokenizer_identity='codec-v1',language=language,
                                memory=memory,attempts=3,update_bounds=UpdateBounds())
        for mode in ('rtt','rtma','joint'):
            symbols=(['c3','c1','c0'] if mode in ('rtt','joint') else [])+(['sub','c1','x0'] if mode in ('rtma','joint') else [])
            parent=parse_rule(language,mode,symbols)
            first=factory.create(parent,0,7,lambda *args:None)
            second=factory.create(parent,1,8,lambda *args:None)
            self.assertNotEqual(first.fast.a.data_ptr(),second.fast.a.data_ptr())
            self.assertIsNot(first.memory,second.memory)
            self.assertEqual(first.parent,parent)
        with torch.no_grad():model.mixer.get_input_embeddings().weight.add_(.01)
        with self.assertRaises(ValueError):factory.create(parent,2,9,lambda *args:None)
    def test_codec_rejects_non_roundtripping_tokens(self):
        class Bad(CodecTokenizer):
            def encode(self,*args,**kwargs):return [1,2]
        with self.assertRaises(ValueError):real_token_codec(Bad(),Language(),'rtt',64)


if __name__=='__main__':unittest.main()
