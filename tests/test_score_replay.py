import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from mmla_rsi.score_replay import prefix_score_gradient,ScoreReplayMismatch

class ReplayTests(unittest.TestCase):
 def fixture(self):
  fast=SimpleNamespace(a=torch.tensor([.1,.2],requires_grad=True),b=torch.tensor([.3,-.2],requires_grad=True))
  prompt=(1,2,3);tokens=(2,1,0);seen=[];back=[]
  def forward(ids,positions):
   self.assertEqual(ids,prompt+tokens[:len(seen)]);self.assertEqual(positions,[len(ids)-1]);seen.append(tuple(ids))
   x=fast.a.sum()+len(ids)*fast.b.sum();return torch.stack((x,-x,2*x))[None]
  def selected(logits,j,token):return logits.float().cpu().log_softmax(0)[token]
  values=[]
  for j,t in enumerate(tokens):
   x=fast.a.sum()+(len(prompt)+j)*fast.b.sum();values.append(torch.stack((x,-x,2*x)).log_softmax(0)[t])
  lp=sum(float(x.detach()) for x in values);ref=torch.autograd.grad(torch.stack(values).sum()*.7,(fast.a,fast.b))
  return fast,prompt,tokens,seen,back,forward,selected,lp,ref
 def test_prefixes_and_exact_sampler_sum_and_score(self):
  f,p,t,seen,back,fw,sel,lp,ref=self.fixture();before=(f.a.detach().clone(),f.b.detach().clone())
  g,e=prefix_score_gradient(f,p,t,forward=fw,selected=sel,loss=.7,sampled_log_probability=lp,tolerance=1e-4,on_backward=back.append)
  self.assertEqual(e['replay_delta'],0.);self.assertEqual(back,[3,4,5]);self.assertEqual(len(seen),3)
  for a,b in zip(g,ref):torch.testing.assert_close(a,b)
  self.assertTrue(torch.equal(before[0],f.a));self.assertTrue(torch.equal(before[1],f.b));self.assertIsNone(f.a.grad);self.assertIsNone(f.b.grad)
 def test_failure_retains_measured_work_and_returns_no_gradient(self):
  f,p,t,seen,back,fw,sel,lp,ref=self.fixture()
  with self.assertRaises(ScoreReplayMismatch) as exc:prefix_score_gradient(f,p,t,forward=fw,selected=sel,loss=.7,sampled_log_probability=lp+.01,tolerance=1e-4,on_backward=back.append)
  self.assertEqual(back,[3,4,5]);self.assertEqual(exc.exception.evidence['backward_calls'],3);self.assertFalse(exc.exception.evidence['temporary_gradients_committed']);self.assertIsNone(f.a.grad)

if __name__=='__main__':unittest.main()
