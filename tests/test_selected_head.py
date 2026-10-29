import copy,dataclasses,unittest
import torch
from mmla_rsi.adapter import AdapterProfile,SelectedHeadProfile,CausalLarcAdapter
from mmla_rsi.reader_binding import FrozenReaderBinding
from tiny_helper import NeuralTests
from test_adapter import CodecTokenizer


class SelectedHeadTests(unittest.TestCase):
    def models(self):
        fixture=NeuralTests();fixture.setUp();full=fixture.m
        with torch.no_grad():full.rho.b.fill_(.02)
        selected=CausalLarcAdapter(copy.deepcopy(full.mixer),SelectedHeadProfile(32,256,8),
            seed=1,substrate_identity={'fixture':True})
        selected.load_state_dict(full.state_dict())
        return full,selected

    def test_logits_input_gradients_probabilities_and_rng(self):
        full,selected=self.models();a=full.rho.begin(9);b=selected.rho.begin(9)
        ids=torch.tensor([[0,1,3,4,5]]);mask=torch.tensor([[0,1,1,1,1]]);positions=torch.tensor([1,4,1])
        rng=torch.get_rng_state().clone()
        x=full.forward_batch(ids,mask,fast_states=[a],logit_positions=positions)
        y=selected.forward_batch(ids,mask,fast_states=[b],logit_positions=positions)
        torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(x.float().log_softmax(-1),y.float().log_softmax(-1),rtol=1e-5,atol=1e-6)
        gx=torch.autograd.grad(x.square().sum(),(a.a,a.b))
        gy=torch.autograd.grad(y.square().sum(),(b.a,b.b))
        for u,v in zip(gx,gy):torch.testing.assert_close(u,v,rtol=2e-5,atol=2e-6)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in selected.mixer.parameters()))

    def test_head_sees_only_selected_rows_and_full_prefix_still_used(self):
        full,selected=self.models();seen=[]
        hook=selected.mixer.lm_head.register_forward_pre_hook(lambda m,args:seen.append(tuple(args[0].shape)))
        ids=torch.tensor([[1,2,3,4,5]])
        selected.forward_batch(ids,torch.ones_like(ids),logit_positions=torch.tensor([4]))
        self.assertEqual(seen,[(1,1,32)]);hook.remove()
        torch.testing.assert_close(full.forward_batch(ids,torch.ones_like(ids)),
            selected.forward_batch(ids,torch.ones_like(ids)),rtol=1e-5,atol=1e-6)

    def test_projection_abi_binds_reader_and_rejects_mode_change(self):
        full,selected=self.models();tokenizer=CodecTokenizer()
        a=FrozenReaderBinding(full,tokenizer,tokenizer_identity='fixture')
        b=FrozenReaderBinding(selected,tokenizer,tokenizer_identity='fixture')
        self.assertNotEqual(a.reader_identity,b.reader_identity)
        selected.profile=AdapterProfile(32,256,8)
        with self.assertRaisesRegex(ValueError,'binding'):b.validate(selected,tokenizer,tokenizer_identity='fixture')

    def test_invalid_projection_contract_rejected(self):
        full,_=self.models()
        with self.assertRaises(ValueError):SelectedHeadProfile(32,256,8,decoder_path='')
        with self.assertRaises(ValueError):SelectedHeadProfile(32,256,8,projection_abi='unknown')
        full.mixer.lm_head=torch.nn.Sequential(full.mixer.lm_head)
        with self.assertRaises(ValueError):CausalLarcAdapter(full.mixer,SelectedHeadProfile(32,256,8),seed=1,substrate_identity={'fixture':True})

if __name__=='__main__':unittest.main()
