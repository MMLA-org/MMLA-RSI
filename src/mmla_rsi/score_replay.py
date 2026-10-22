"""Replay each sampled conditional with the same prefix and one output row.

Temporary gradients are accumulated without retaining earlier graphs. A failed
probability check never returns them to an updater; actual attempted work remains
charged. No cache, sampling filter, precision change or tolerance relaxation.
"""
import math
import torch


class ScoreReplayMismatch(FloatingPointError):
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__('sample-prefix replay probability mismatch: '+repr(evidence))


def prefix_score_gradient(fast,prompt,tokens,*,forward,selected,loss,
                          sampled_log_probability,tolerance,on_backward):
    if not tokens:raise ValueError('nonempty issued rollout required')
    gradients=[torch.zeros_like(fast.a),torch.zeros_like(fast.b)]
    logp=0.;lengths=[]
    for j,token in enumerate(tokens):
        ids=prompt+tokens[:j];lengths.append(len(ids))
        logits=forward(ids,[len(ids)-1])[0]
        value=selected(logits,j,token)
        if value.numel()!=1 or not bool(torch.isfinite(value)):
            raise FloatingPointError('nonfinite prefix score replay')
        # Same Python-double, left-to-right sum as the actual sampler.
        logp+=float(value.detach())
        on_backward(len(ids))
        pair=torch.autograd.grad(value*float(loss),(fast.a,fast.b))
        for target,g in zip(gradients,pair):
            if not bool(torch.isfinite(g).all()):raise FloatingPointError('nonfinite prefix score gradient')
            target.add_(g.detach())
        del logits,value,pair
    delta=logp-sampled_log_probability
    evidence=dict(log_probability=logp,sampled_log_probability=sampled_log_probability,
        replay_delta=delta,tolerance=tolerance,replay='sample-prefix-single-position-v1',
        forward_prefix_lengths=lengths,backward_calls=len(tokens),temporary_gradients_committed=False,
        estimator='zero-baseline loss-times-score')
    if not math.isfinite(delta) or abs(delta)>tolerance:raise ScoreReplayMismatch(evidence)
    return tuple(gradients),evidence
