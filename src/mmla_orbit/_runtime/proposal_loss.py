"""Task-outcome-targeted slow proposal-rho objective, no runtime adaptation.

The input uses the actual deployment header and parent-managed M. Teacher
programs receive summed grammar-conditional token NLL, averaged over examples.
No gradient is claimed through discrete task execution or target selection.
"""
from recipe import EXAMPLES,PRESENTATIONS,UPDATES
from dataclasses import dataclass
import copy
import math
import re
import torch

@dataclass(frozen=True)
class OptimizerSpec:
    learning_rate: float = 1e-4
    betas: tuple = (.9,.999)
    eps: float = 1e-8
    weight_decay: float = 0.
    clip: float = 1.
    batch_size: int = 8
    epochs: int = 2
    examples: int = EXAMPLES

def example_logprobs(model,example,*,on_work):
    prompt=example['prompt'];tokens=example['tokens'];masks=example['masks']
    if (type(prompt) is not list or type(tokens) is not list or type(masks) is not list
            or not prompt or not tokens or len(tokens)!=len(masks)
            or len(prompt)+len(tokens)>model.profile.local_tokens
            or any(type(t) is not int or t<0 for t in prompt+tokens)
            or any(type(mask) is not list or not mask
                   or any(type(t) is not int or t<0 for t in mask)
                   or len(mask)!=len(set(mask)) or tok not in mask for tok,mask in zip(tokens,masks))):
        raise ValueError('complete causal grammar-supervised example required')
    device=model.rho.a.device
    # Last target token is predicted, never a needed input. One episode per
    # forward avoids padding and cross-episode state or arithmetic coupling.
    ids=torch.tensor([prompt+tokens[:-1]],device=device,dtype=torch.long)
    pos=torch.arange(len(prompt)-1,len(prompt)+len(tokens)-1,device=device,dtype=torch.long)
    on_work(dict(kind='proposal_training_forward',input_tokens=ids.numel(),target_tokens=len(tokens)))
    logits=model.forward_batch(ids,torch.ones_like(ids),fast_states=None,memory=None,logit_positions=pos)[0]
    if logits.ndim!=2 or logits.shape[0]!=len(tokens):raise ValueError('causal target positions mismatch')
    distributions=[]
    for j,(token,mask) in enumerate(zip(tokens,masks)):
        scores=logits[j,mask].float()
        if not torch.isfinite(scores).all():raise FloatingPointError('nonfinite grammar logits')
        distributions.append(scores.log_softmax(0))
    return distributions

def example_nll(model,example,*,on_work):
    values=example_logprobs(model,example,on_work=on_work)
    loss=torch.stack([-v[mask.index(token)] for v,token,mask in
        zip(values,example['tokens'],example['masks'])]).sum()
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite full-program NLL')
    return loss

class ProposalTrainer:
    def __init__(self,model,*,data_identity,spec=OptimizerSpec()):
        if (spec!=OptimizerSpec() or type(data_identity) is not str
                or re.fullmatch('[0-9a-f]{64}',data_identity) is None):
            raise ValueError('fixed finite proposal training contract required')
        self.model,self.spec,self.data_identity=model,spec,data_identity
        self.steps=0;self.cursor=0;self.poisoned=False;self.finished=False
        for name,p in model.named_parameters():
            p.requires_grad_(name in ('rho.a','rho.b'));p.grad=None
        model.eval()
        self.parameters=(model.rho.a,model.rho.b)
        if tuple(id(p) for p in model.parameters() if p.requires_grad)!=tuple(map(id,self.parameters)):
            raise ValueError('proposal rho A/B must be the only trainable parameters')
        self.optimizer=torch.optim.AdamW(self.parameters,lr=spec.learning_rate,betas=spec.betas,
            eps=spec.eps,weight_decay=0.,foreach=False,fused=False)
        self.initial_version=int(model.rho.version)
        if self.initial_version!=128:raise ValueError('original proposal rho128 required')
        self.profile_identity=model.profile.identity
        self.substrate=copy.deepcopy(model.substrate_identity)
        self.statuses=copy.deepcopy(model.module_status)
        self.initialization_guard=self._initialization_guard()
        self.frozen=tuple((n,id(p),p._version,p.data_ptr()) for n,p in
            tuple(model.named_parameters())+tuple(model.named_buffers()) if n not in ('rho.a','rho.b','rho.version'))

    def _initialization_guard(self):
        init=self.model.rho.initialization
        return (init['seed'],tuple((k,id(init[k]),init[k]._version,init[k].data_ptr()) for k in ('a','b')))

    def check(self):
        if self.poisoned:raise RuntimeError('failed training state is not reusable')
        current=tuple((n,id(p),p._version,p.data_ptr()) for n,p in
            tuple(self.model.named_parameters())+tuple(self.model.named_buffers()) if n not in ('rho.a','rho.b','rho.version'))
        statuses={**self.statuses,**({'rho':'TRAINED_TASK_OUTCOME_MANAGER_PROPOSAL'} if self.finished else {})}
        if (current!=self.frozen or any(m.training for m in self.model.modules())
                or self.model.profile.identity!=self.profile_identity or self.model.substrate_identity!=self.substrate
                or self.model.module_status!=statuses or self._initialization_guard()!=self.initialization_guard
                or tuple(id(p) for p in self.model.parameters() if p.requires_grad)!=(
                    () if self.finished else tuple(map(id,self.parameters)))
                or int(self.model.rho.version)!=self.initial_version+self.steps):
            raise RuntimeError('task/shared modules, training ownership or rho version changed')

    def step(self,examples,*,offset,on_work):
        self.check()
        if (self.finished or len(examples)!=8 or offset!=self.cursor or self.steps>=UPDATES
                or len({x['identity'] for x in examples})!=8):
            raise ValueError('fixed eight-example schedule/cursor required')
        try:
            self.optimizer.zero_grad(set_to_none=True);loss=0.
            for example in examples:
                value=example_nll(self.model,example,on_work=on_work)/8
                value.backward();loss+=float(value.detach())
                on_work(dict(kind='proposal_training_backward',examples=1))
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.parameters):
                raise FloatingPointError('rho did not receive finite task-target gradients')
            norm=torch.nn.utils.clip_grad_norm_(self.parameters,self.spec.clip,error_if_nonfinite=True)
            self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in self.parameters):
                raise FloatingPointError('nonfinite slow residual update')
            for state in self.optimizer.state.values():
                if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):
                    raise FloatingPointError('nonfinite optimizer state')
            with torch.no_grad():self.model.rho.version.add_(1)
            self.steps+=1;self.cursor+=8;self.optimizer.zero_grad(set_to_none=True);self.check()
            return dict(step=self.steps,next_example=self.cursor,program_nll=loss,
                unclipped_gradient_norm=float(norm),rho_version=int(self.model.rho.version),
                online_numerical_updates=0,objective='FIT_CONDITIONAL_MIXTURE_RISK_PROGRAM_MASKED_SEQUENCE_NLL')
        except BaseException:
            self.poisoned=True
            raise

    def export(self):
        self.check()
        if any(p.grad is not None for p in self.parameters):raise ValueError('quiet checkpoint required')
        return dict(schema='orbit-outcome-proposal-trainer-v1',data_identity=self.data_identity,
            spec=self.spec.__dict__.copy(),steps=self.steps,cursor=self.cursor,
            initial_version=self.initial_version,rho=copy.deepcopy(dict(self.model.rho.state_dict())),
            initialization=copy.deepcopy(self.model.rho.initialization),finished=self.finished,
            optimizer=copy.deepcopy(self.optimizer.state_dict()),torch_rng=torch.get_rng_state().clone(),
            cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()] if self.model.rho.a.is_cuda else [])

    def restore(self,state):
        """Restore verified decoded state into an unstarted original-rho trainer.

        Source/data identity and serialized bytes are the host's responsibility;
        this method validates all numeric state before mutating live parameters.
        """
        self.check()
        if (self.steps or self.cursor or self.finished or state.get('schema')!='orbit-outcome-proposal-trainer-v1'
                or set(state)!={'schema','data_identity','spec','steps','cursor','initial_version','rho',
                                'initialization','finished','optimizer','torch_rng','cuda_rng'}
                or state['data_identity']!=self.data_identity or state['spec']!=self.spec.__dict__
                or state['initial_version']!=self.initial_version
                or type(state['steps']) is not int or not 0<=state['steps']<=UPDATES
                or type(state['cursor']) is not int or state['cursor']!=8*state['steps']
                or type(state['finished']) is not bool or (state['finished'] and state['steps']!=UPDATES)):
            raise ValueError('fresh exact source/data/schedule-bound trainer required')
        r=state['rho'];old=self.model.rho.state_dict()
        if set(r)!=set(old):raise ValueError('complete rho state required')
        for name,value in old.items():
            if (not torch.is_tensor(r[name]) or r[name].shape!=value.shape or r[name].dtype!=value.dtype
                    or not torch.isfinite(r[name]).all()):
                raise ValueError('rho tensor layout/numeric mismatch')
        if int(r['version'])!=self.initial_version+state['steps']:
            raise ValueError('rho and optimizer version mismatch')
        init=state['initialization'];expected=self.model.rho.initialization
        if (set(init)!=set(expected) or init['seed']!=expected['seed']
                or any(not torch.is_tensor(init[k]) or init[k].dtype!=expected[k].dtype
                       or init[k].shape!=expected[k].shape or not torch.equal(init[k].cpu(),expected[k].cpu())
                       for k in ('a','b'))):
            raise ValueError('original initialization cannot be replaced by trained factors')
        # Validate RAW serialized moments before load_state_dict can cast them
        # to parameter dtype/device or reorder the optimizer's parameter IDs.
        opt=state['optimizer'];expected_opt=self.optimizer.state_dict()
        if (type(opt) is not dict or set(opt)!={'state','param_groups'}
                or opt['param_groups']!=expected_opt['param_groups']
                or type(opt['state']) is not dict):
            raise ValueError('fixed optimizer configuration/order changed')
        ids=expected_opt['param_groups'][0]['params']
        if set(opt['state'])!=(set(ids) if state['steps'] else set()):
            raise ValueError('complete optimizer inventory required')
        for index,param in zip(ids,self.parameters):
            slots=opt['state'].get(index,{})
            if not state['steps']:
                if slots:raise ValueError('unstarted optimizer has state')
                continue
            if set(slots)!={'step','exp_avg','exp_avg_sq'}:
                raise ValueError('complete AdamW moments required')
            if (not torch.is_tensor(slots['step']) or slots['step'].shape!=torch.Size([])
                    or slots['step'].dtype!=torch.float32 or slots['step'].device.type!='cpu'
                    or float(slots['step'])!=state['steps']):raise ValueError('AdamW cursor mismatch')
            for k in ('exp_avg','exp_avg_sq'):
                if (not torch.is_tensor(slots[k]) or slots[k].dtype!=param.dtype
                        or slots[k].shape!=param.shape or not torch.isfinite(slots[k]).all()):
                    raise ValueError('AdamW moment layout/numeric mismatch')
            if (slots['exp_avg_sq']<0).any():raise ValueError('negative AdamW second moment')
        if not torch.is_tensor(state['torch_rng']) or state['torch_rng'].device.type!='cpu':
            raise ValueError('CPU RNG bytes required')
        torch.Generator(device='cpu').set_state(state['torch_rng'])
        expected_devices=torch.cuda.device_count() if self.model.rho.a.is_cuda else 0
        if len(state['cuda_rng'])!=expected_devices:raise ValueError('complete device RNG required')
        for i,rng in enumerate(state['cuda_rng']):torch.Generator(device='cuda:'+str(i)).set_state(rng)
        backup=self.export()
        try:
            self.model.rho.load_state_dict(r,strict=True)
            self.optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['torch_rng'])
            if expected_devices:torch.cuda.set_rng_state_all(state['cuda_rng'])
            self.steps=state['steps'];self.cursor=state['cursor'];self.finished=state['finished']
            if self.finished:
                for p in self.model.parameters():p.requires_grad_(False)
                self.model.module_status['rho']='TRAINED_TASK_OUTCOME_MANAGER_PROPOSAL'
            self.check()
        except BaseException as error:
            try:
                self.model.rho.load_state_dict(backup['rho'],strict=True)
                self.optimizer.load_state_dict(backup['optimizer'])
                torch.set_rng_state(backup['torch_rng'])
                if expected_devices:torch.cuda.set_rng_state_all(backup['cuda_rng'])
                self.steps=0;self.cursor=0;self.finished=False
                for p in self.parameters:p.requires_grad_(True)
                self.model.module_status=copy.deepcopy(self.statuses)
                self.check()
            except BaseException as rollback:
                self.poisoned=True;error.add_note('restore rollback failed: '+repr(rollback))
            raise

    def finish(self):
        self.check()
        if self.steps!=UPDATES or self.cursor!=PRESENTATIONS:raise RuntimeError('complete two-epoch curriculum required')
        self.finished=True
        for p in self.model.parameters():p.requires_grad_(False);p.grad=None
        self.model.module_status['rho']='TRAINED_TASK_OUTCOME_MANAGER_PROPOSAL'
        return dict(steps=self.steps,examples=self.cursor,rho_version=int(self.model.rho.version),
                    online_Phi_updates=0,task_reader_changed=False,shared_modules_changed=False)
