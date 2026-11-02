import copy,unittest
import torch
from mmla_rsi.adapter import CausalLarcAdapter,RecomputedDecoderProfile
from mmla_rsi.reader_binding import FrozenReaderBinding
import test_selected_head
from test_adapter import CodecTokenizer

class RecomputedDecoderTests(unittest.TestCase):
    def test_logits_fast_gradients_and_actual_replay(self):
        _,a=test_selected_head.SelectedHeadTests().models()
        b=CausalLarcAdapter(copy.deepcopy(a.mixer),RecomputedDecoderProfile(32,256,8),seed=1,substrate_identity={'fixture':True})
        b.load_state_dict(a.state_dict());fast_a=a.rho.begin(9);fast_b=b.rho.begin(9)
        ids=torch.tensor([[0,1,3,4,5]]);mask=torch.tensor([[0,1,1,1,1]]);positions=torch.tensor([1,4,1])
        entries=[];hook=b.mixer.model.register_forward_pre_hook(lambda module,args:entries.append(1))
        rng=torch.get_rng_state().clone()
        x=a.forward_batch(ids,mask,fast_states=[fast_a],logit_positions=positions)
        y=b.forward_batch(ids,mask,fast_states=[fast_b],logit_positions=positions)
        torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
        gx=torch.autograd.grad(x.square().sum(),(fast_a.a,fast_a.b))
        gy=torch.autograd.grad(y.square().sum(),(fast_b.a,fast_b.b))
        for u,v in zip(gx,gy):torch.testing.assert_close(u,v,rtol=2e-5,atol=2e-6)
        self.assertEqual(len(entries),2);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in b.mixer.parameters()))
        entries.clear()
        with torch.no_grad():z=b.forward_batch(ids,mask,fast_states=[fast_b],logit_positions=positions)
        torch.testing.assert_close(z,y);self.assertEqual(len(entries),1);hook.remove()
        tokenizer=CodecTokenizer();old=FrozenReaderBinding(a,tokenizer,tokenizer_identity='fixture')
        new=FrozenReaderBinding(b,tokenizer,tokenizer_identity='fixture')
        self.assertNotEqual(old.reader_identity,new.reader_identity)
        b.profile=a.profile
        with self.assertRaisesRegex(ValueError,'binding'):new.validate(b,tokenizer,tokenizer_identity='fixture')

    def test_unknown_recomputation_abi_rejected(self):
        with self.assertRaises(ValueError):RecomputedDecoderProfile(32,256,8,activation_recompute_abi='unknown')

if __name__=='__main__':unittest.main()
