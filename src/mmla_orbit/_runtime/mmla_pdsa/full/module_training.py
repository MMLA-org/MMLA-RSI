"""Finite module training, distinct from rho's first-order meta objective.

The callback returns one scalar per complete semantic task. It cannot advance
samplers, mutate model/optimizer state or use another split. The fixed schedule
is reconstructed from (schedule_identity, committed steps), so failure consumes
compute but never silently skips a data batch.
"""
import copy
import random
import torch
from .numerics import all_finite
from .local_rng import capture_cuda,validate_cuda,restore_cuda
from .distributed import EpisodeParallel
from .neural import tensor_digest
from .profile import digest
from .training import optimizer_digest

GROUPS={
 'frontend':('frontend',),
 'memory':('consolidator','canonical_key','canonical_value','readout'),
 'constructor':('constructor',),
 'risk':('risk',),
 'hybrid':('hybrid',),
 'joint_risk':('joint_risk',),
}
STATUS={'frontend':('frontend',),'memory':('consolidation','canonical_encoder','readout'),
        'constructor':('constructor',),'risk':('risk',),'hybrid':('hybrid',),'joint_risk':('joint_risk',)}


def finite_optimizer(optimizer):
    values=(value for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))
    if not all_finite(values):raise FloatingPointError('nonfinite optimizer slot')


def rng_snapshot():
    return dict(torch=torch.get_rng_state(),python=random.getstate(),
                cuda=capture_cuda())


def restore_rng(state):
    # Validate every destination before publishing any global RNG stream.
    torch.Generator(device='cpu').set_state(state['torch'])
    random.Random().setstate(state['python']);validate_cuda(state['cuda'])
    restore_cuda(state['cuda'])
    torch.set_rng_state(state['torch']);random.setstate(state['python'])


class ModuleTrainer:
    kind='mmla-module-trainer-v1'
    def __init__(self,model,*,stage,learning_rate,max_steps,global_batch,microbatch,
                 schedule_identity,data_identity,feature_identity,clip=1.,parallel=None):
        if stage not in GROUPS or not 0<learning_rate<=1 or not 0<clip<1e6:
            raise ValueError('registered module stage and bounded optimization')
        if (type(max_steps) is not int or not 1<=max_steps<=4096 or type(global_batch) is not int
                or not 1<=global_batch<=64 or type(microbatch) is not int or not 1<=microbatch<=global_batch):
            raise ValueError('finite steps and exact real-task batch required')
        for identity in (schedule_identity,data_identity,feature_identity):
            if type(identity) is not str or len(identity)!=64:raise ValueError('exact pipeline identities required')
            int(identity,16)
        self.model,self.stage=model,stage;self.parallel=parallel or EpisodeParallel()
        self.max_steps,self.global_batch,self.microbatch=max_steps,global_batch,microbatch
        self.clip,self.steps,self.nonzero_steps=clip,0,0
        prefixes=tuple(x+'.' for x in GROUPS[stage])
        self.named=tuple((n,p) for n,p in model.named_parameters() if n.startswith(prefixes))
        if not self.named:raise ValueError('empty module stage')
        names={n for n,_ in self.named}
        for name,parameter in model.named_parameters():
            parameter.requires_grad_(name in names);parameter.grad=None
        self.optimizer=torch.optim.AdamW([p for _,p in self.named],lr=learning_rate,
            betas=(.9,.999),eps=1e-8,weight_decay=0.)
        self.contract=dict(kind=self.kind,stage=stage,schedule=schedule_identity,data=data_identity,
            features=feature_identity,max_steps=max_steps,global_batch=global_batch,
            microbatch=microbatch,clip=clip,learning_rate=learning_rate,
            betas=[.9,.999],eps=1e-8,weight_decay=0.,parameters=[n for n,_ in self.named],
            profile=model.profile.identity,substrate=copy.deepcopy(model.substrate_identity))
        self.contract_identity=digest(self.contract)

    def verify_binding(self,model):
        params=[p for group in self.optimizer.param_groups for p in group['params']]
        expected=tuple((n,p) for n,p in model.named_parameters() if n.startswith(tuple(x+'.' for x in GROUPS[self.stage])))
        if model is not self.model or len(params)!=len(expected) or len(expected)!=len(self.named):
            raise ValueError('complete actual model module binding required')
        for (name,param),(other_name,other),optim in zip(self.named,expected,params):
            if name!=other_name or param is not other or optim is not param:raise ValueError('module optimizer binding changed')
        if (digest(self.contract)!=self.contract_identity or model.profile.identity!=self.contract['profile']
                or model.substrate_identity!=self.contract['substrate']):raise ValueError('frozen module contract changed')
        actual=dict(stage=self.stage,max_steps=self.max_steps,global_batch=self.global_batch,
                    microbatch=self.microbatch,clip=self.clip)
        if any(value!=self.contract[name] for name,value in actual.items()):raise ValueError('active trainer configuration changed')
        group=self.optimizer.param_groups[0]
        expected=dict(lr=self.contract['learning_rate'],betas=tuple(self.contract['betas']),
                      eps=self.contract['eps'],weight_decay=0.,amsgrad=False,maximize=False)
        if len(self.optimizer.param_groups)!=1 or any(group[name]!=value for name,value in expected.items()):
            raise ValueError('active optimizer configuration changed')


    def export_state(self):
        self.verify_binding(self.model);finite_optimizer(self.optimizer)
        return dict(kind=self.kind,contract=copy.deepcopy(self.contract),steps=self.steps,
            next_batch=self.steps,nonzero_steps=self.nonzero_steps,optimizer=self.optimizer.state_dict())

    def validate_state(self,state):
        if (set(state)!={'kind','contract','steps','next_batch','nonzero_steps','optimizer'}
                or state['kind']!=self.kind or state['contract']!=self.contract
                or type(state['steps']) is not int or not 0<=state['steps']<=self.max_steps
                or type(state['next_batch']) is not int or state['next_batch']!=state['steps']
                or type(state['nonzero_steps']) is not int or not 0<=state['nonzero_steps']<=state['steps']):
            raise ValueError('exact module stage/schedule/optimizer contract required')
        candidate=copy.deepcopy(self.optimizer);candidate.load_state_dict(state['optimizer'])
        finite_optimizer(candidate)
        for group,expected in zip(candidate.param_groups,self.optimizer.param_groups):
            for key in ('lr','betas','eps','weight_decay','amsgrad','maximize'):
                if group[key]!=expected[key]:raise ValueError('saved optimizer hyperparameter changed')
            for p in group['params']:
                s=candidate.state.get(p,{})
                if state['steps'] and set(s)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('complete Adam state required')
                if s:
                    if s['step'].numel()!=1 or float(s['step'])!=state['steps']:raise ValueError('Adam cursor mismatch')
                    if any(s[k].shape!=p.shape or s[k].dtype!=p.dtype for k in ('exp_avg','exp_avg_sq')):
                        raise ValueError('Adam moment shape/dtype mismatch')
        return candidate

    def import_state(self,state):
        self.validate_state(state)
        self.optimizer.load_state_dict(state['optimizer'])
        self.steps,self.nonzero_steps=state['steps'],state['nonzero_steps']

    def update(self,tasks,loss_batch):
        params=tuple(p for _,p in self.named);device=params[0].device;parallel=self.parallel
        error=None
        try:
            self.verify_binding(self.model)
            if self.steps>=self.max_steps:raise RuntimeError('finite module steps exhausted')
            groups=[x.group for x in tasks]
            if any(type(g) is not str or not g for g in groups):raise ValueError('semantic groups required')
            if any(not p.requires_grad or hasattr(p,'main_grad') for p in params):raise ValueError('module ownership changed')
            finite_optimizer(self.optimizer)
            batch_contract=dict(contract=self.contract_identity,step=self.steps,optimizer=optimizer_digest(self.optimizer.state_dict(),tensor_values=parallel.size>1))
        except Exception as exc:error=exc
        if not parallel.agree(error is None,device=device):raise ValueError('module preflight failed') from error
        parallel.verify_batch(groups,batch_contract)
        if sum(parallel.gather(len(tasks)))!=self.global_batch:raise ValueError('exact global real-task count required')
        if parallel.size>1:parallel.verify_parameters(params)
        before={n:p.detach().clone() for n,p in self.named};state=copy.deepcopy(self.export_state())
        rng=rng_snapshot();statuses=dict(self.model.module_status)
        joint_updates=int(self.model.joint_risk.updates) if self.stage=='joint_risk' else None
        self.optimizer.zero_grad(set_to_none=True);losses=[]
        # Both the memory encoder and frontend contain a one-layer zero-dropout
        # GRU. cuDNN needs training-mode reserve space for their backward pass.
        # Select recurrent modules by actual stage-owned parameters, not by one
        # stage name; never toggle the frozen mixer or any unowned recurrent.
        owned={id(p) for p in params}
        recurrent_modes=[(m,m.training) for m in self.model.modules()
            if isinstance(m,torch.nn.RNNBase) and any(id(p) in owned for p in m.parameters())]
        if any(m.dropout!=0 for m,_ in recurrent_modes):raise ValueError('deterministic zero-dropout recurrent training required')
        for module,_ in recurrent_modes:module.train(True)
        try:
            error=None
            try:
                for start in range(0,len(tasks),self.microbatch):
                    batch=tasks[start:start+self.microbatch];terms=loss_batch(batch)
                    if terms.ndim!=1 or len(terms)!=len(batch) or not torch.isfinite(terms).all():
                        raise ValueError('one finite task-weighted loss per actual episode')
                    terms.sum().backward();losses.append(terms.detach())
                if not all_finite(p.grad for p in params):
                    raise FloatingPointError('nonfinite module gradient')
            except Exception as exc:error=exc
            if not parallel.agree(error is None,device=device):raise RuntimeError('module batch aborted') from error
            total=parallel.reduce_task_sums(params,local_tasks=len(tasks))
            error=None;norm=None
            try:
                losses=torch.cat(losses).tolist() if losses else []
                if not all_finite(p.grad for p in params):raise FloatingPointError('nonfinite reduced module gradient')
                norm=torch.nn.utils.clip_grad_norm_(params,self.clip,error_if_nonfinite=True)
                if float(norm)==0 and self.nonzero_steps==0:raise FloatingPointError('stage has no task gradient signal')
                self.optimizer.step();finite_optimizer(self.optimizer)
                if not all_finite(params):raise FloatingPointError('nonfinite module successor')
            except Exception as exc:error=exc
            if not parallel.agree(error is None,device=device):raise RuntimeError('module optimizer commit aborted') from error
            if parallel.size>1:parallel.verify_parameters(params)
            states=parallel.gather(optimizer_digest(self.optimizer.state_dict(),tensor_values=parallel.size>1))
            if len(set(states))!=1:raise RuntimeError('module optimizer successors differ')
            self.steps+=1;self.nonzero_steps+=int(float(norm)>0)
            if self.stage=='joint_risk':self.model.joint_risk.updates.add_(1)
            for key in STATUS[self.stage]:
                self.model.module_status[key]='TRAINED_UNCALIBRATED' if self.stage in ('risk','joint_risk') else 'TRAINED_BY_DECLARED_MODULE_OBJECTIVE'
            return dict(stage=self.stage,step=self.steps,global_tasks=total,task_losses=losses,
                        mean_gradient_norm=float(norm),microbatch=self.microbatch)
        except Exception:
            with torch.no_grad():
                for name,p in self.named:p.copy_(before[name])
            self.import_state(state);restore_rng(rng);self.model.module_status=statuses
            if joint_updates is not None:self.model.joint_risk.updates.fill_(joint_updates)
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            for module,mode in recurrent_modes:module.train(mode)
