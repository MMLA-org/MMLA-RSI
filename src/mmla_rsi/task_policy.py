"""Full-vocabulary task actions and their actual score-function gradients.

Sampling is stochastic over the whole vocabulary. EOS and the fixed token cap
define the action law; no grammar survivor filtering or teacher forcing occurs.
The only input is the immutable task request and current authoritative memory.
"""
from dataclasses import dataclass,asdict,replace
import math,time
import torch
from .neural import tensor_digest
from .rules import identity


@dataclass(frozen=True)
class TaskSampling:
    eos_ids: tuple[int,...]
    max_tokens: int
    header_tokens: int
    memory_tokens: int
    temperature: float=1.
    replay_tolerance: float=1e-4
    def __post_init__(self):
        if (type(self.eos_ids) is not tuple or not self.eos_ids
                or len(set(self.eos_ids))!=len(self.eos_ids)
                or any(type(x) is not int or x<0 for x in self.eos_ids)
                or any(type(x) is not int or x<1 for x in (self.max_tokens,self.header_tokens,self.memory_tokens))
                or not 0<self.temperature<math.inf or not 0<=self.replay_tolerance<math.inf):
            raise ValueError('fixed bounded full-vocabulary sampling law required')


@dataclass(frozen=True)
class TaskRollout:
    tokens: tuple
    prompt: tuple
    header: str
    action: str
    memory_text: str
    memory_read: dict
    policy_read: str
    sampling: str
    log_probability: float
    rng_before: torch.Tensor
    rng_after: torch.Tensor
    stop: str
    cost: dict
    seal: str=''
    def digest(self):
        data=asdict(self);data.pop('seal')
        for k in ('rng_before','rng_after'):data[k]=tensor_digest({k:getattr(self,k)})
        return identity(data)


class TaskSamplingFailure(RuntimeError):
    def __init__(self,rollout):
        super().__init__(rollout.stop);self.rollout=rollout


class TaskSamplingInterrupted(BaseException):
    def __init__(self,rollout,cause):
        super().__init__(type(cause).__name__)
        self.rollout,self.cause=rollout,cause


class TaskPolicy:
    def __init__(self,model,tokenizer,*,sampling,tokenizer_identity):
        if not tokenizer_identity:raise ValueError('exact tokenizer identity required')
        self.model,self.tokenizer,self.sampling=model,tokenizer,sampling
        self.tokenizer_identity=tokenizer_identity;self._tokenizer=tokenizer
        vocab=model.mixer.get_input_embeddings().weight.shape[0]
        if (max(sampling.eos_ids)>=vocab or sampling.max_tokens>model.profile.max_new_tokens
                or sampling.header_tokens+sampling.memory_tokens+sampling.max_tokens+32>model.profile.local_tokens):
            raise ValueError('declared task input/action does not fit exact model')
        self.vocab=vocab;self._guarded=self._guard()
        self._reader_definition=self._reader_config()
        self.reader=identity(dict(configuration=self._reader_definition,parameters=tensor_digest(model.state_dict()),profile=asdict(model.profile),
            substrate=model.substrate_identity,tokenizer=tokenizer_identity,torch=torch.__version__))
        self.identity=self._configuration()
    def _reader_config(self):
        return identity(dict(profile=asdict(self.model.profile),substrate=self.model.substrate_identity,
            mixer_config=self.model.mixer.config.to_dict()))
    def _guard(self):
        return tuple((n,id(x),x._version,str(x.device),str(x.dtype),tuple(x.shape),x.data_ptr())
            for n,x in list(self.model.named_parameters())+list(self.model.named_buffers()))
    def _configuration(self):
        return identity(dict(schema='rsi-task-full-vocabulary-v1',sampling=asdict(self.sampling),
                             reader=self.reader,tokenizer=self.tokenizer_identity,vocab=self.vocab))
    def check(self):
        if (any(m.training for m in self.model.modules()) or self._reader_config()!=self._reader_definition
                or self._guard()!=self._guarded or self.tokenizer is not self._tokenizer
                or self._configuration()!=self.identity or torch.is_autocast_enabled('cpu')
                or torch.is_autocast_enabled('cuda')):
            raise ValueError('bound task reader/sampling configuration changed')
    def prompt(self,header,memory_text):
        if type(header) is not str or not header or len(header.encode())>32768:
            raise ValueError('finite immutable task request required')
        if len(self.tokenizer.encode(header,add_special_tokens=False))>self.sampling.header_tokens:
            raise ValueError('task request token allowance exceeded')
        if len(self.tokenizer.encode(memory_text,add_special_tokens=False))>self.sampling.memory_tokens:
            raise ValueError('complete task memory exceeds allowance; no truncation')
        ids=tuple(self.tokenizer.encode(header+'\nMEMORY\n'+memory_text+'\nACTION\n',add_special_tokens=False))
        if (not ids or len(ids)+self.sampling.max_tokens>self.model.profile.local_tokens
                or any(type(x) is not int or not 0<=x<self.vocab for x in ids)):
            raise ValueError('complete task prompt and action must fit')
        return ids
    def _forward(self,ids,fast,positions,meter):
        meter.forward(len(ids),len(positions))
        device=self.model.rho.a.device;value=torch.tensor([ids],device=device,dtype=torch.long)
        return self.model.forward_batch(value,torch.ones_like(value),fast_states=[fast],memory=None,
            logit_positions=torch.tensor(positions,device=device,dtype=torch.long))[0]
    def _decode(self,tokens):
        body=tokens[:-1] if tokens and tokens[-1] in self.sampling.eos_ids else tokens
        return self.tokenizer.decode(list(body),skip_special_tokens=False)
    def sample(self,fast,memory,header,*,meter):
        self.check();self.model.rho.factors(fast)
        text,read=memory.read();prompt=self.prompt(header,text)
        before=fast.rng_state.clone();rng=torch.Generator(device='cpu');rng.set_state(before)
        tokens=[];logp=0.;stop='TOKEN_CAP';calls=0;spent=0;failure=None;action=''
        started=time.monotonic()
        try:
            with torch.no_grad():
                for _ in range(self.sampling.max_tokens):
                    ids=prompt+tuple(tokens);calls+=1;spent+=len(ids)
                    scores=self._forward(ids,fast,[len(ids)-1],meter)[0].float().cpu()/self.sampling.temperature
                    if not torch.isfinite(scores).all():raise FloatingPointError('nonfinite task logits')
                    probs=scores.log_softmax(0);token=int(torch.multinomial(probs.exp(),1,generator=rng))
                    tokens.append(token);logp+=float(probs[token])
                    if token in self.sampling.eos_ids:stop='EOS';break
            action=self._decode(tokens)
            if type(action) is not str:raise ValueError('task tokenizer returned nontext action')
        except BaseException as exc:failure=exc;stop='OPERATIONAL:'+type(exc).__name__
        finally:fast.rng_state=rng.get_state()
        result=TaskRollout(tuple(tokens),prompt,header,action,text,read,fast.identity,self.identity,logp,
            before,fast.rng_state.clone(),stop,dict(forward_calls_attempted=calls,input_tokens_attempted=spent,
                                                  wall_seconds=time.monotonic()-started))
        result=replace(result,seal=result.digest())
        if failure is not None:
            if isinstance(failure,Exception):raise TaskSamplingFailure(result) from failure
            raise TaskSamplingInterrupted(result,failure) from failure
        return result
    def gradient(self,fast,rollout,loss,*,meter):
        self.check();self.model.rho.factors(fast)
        if (type(rollout) is not TaskRollout or rollout.digest()!=rollout.seal
                or rollout.sampling!=self.identity or rollout.policy_read!=fast.identity
                or type(loss) not in (int,float) or not 0<=loss<=1
                or rollout.prompt!=self.prompt(rollout.header,rollout.memory_text)
                or not 1<=len(rollout.tokens)<=self.sampling.max_tokens
                or any(type(x) is not int or not 0<=x<self.vocab for x in rollout.tokens)
                or any(x in self.sampling.eos_ids for x in rollout.tokens[:-1])):
            raise ValueError('arrived feedback must bind the actual task action')
        expected='EOS' if rollout.tokens[-1] in self.sampling.eos_ids else 'TOKEN_CAP'
        if (rollout.stop!=expected or (expected=='TOKEN_CAP' and len(rollout.tokens)!=self.sampling.max_tokens)
                or self._decode(rollout.tokens)!=rollout.action):raise ValueError('task stopping/decoding changed')
        from .score_replay import prefix_score_gradient
        def selected(logits,j,token):
            scores=logits.float().cpu()/self.sampling.temperature
            if not torch.isfinite(scores).all():raise FloatingPointError('nonfinite task replay')
            return scores.log_softmax(0)[token]
        return prefix_score_gradient(fast,rollout.prompt,rollout.tokens,
            forward=lambda ids,pos:self._forward(ids,fast,pos,meter),selected=selected,
            loss=loss,sampled_log_probability=rollout.log_probability,
            tolerance=self.sampling.replay_tolerance,on_backward=meter.backward)
