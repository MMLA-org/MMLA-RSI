"""Complete-demonstration objective, distinct from the old fixed-horizon core.

Dataset identity and lawful recorded feedback belong to the admitted trainer.
This module starts no run. It streams action losses from independent fast leaves
using a declared first-order identity transfer to rho and direct hybrid gradients.
"""
import copy
import torch
from mmla_pdsa.full.outcome_training import OutcomeUpdateCore
from mmla_pdsa.full.profile import digest


class DemonstrationCore(OutcomeUpdateCore):
    def __init__(self, model, config, *, floor_sizes):
        super().__init__(model, config)
        if (type(floor_sizes) is not dict or not floor_sizes or len(floor_sizes)>1024
                or any(type(g) is not str or not g or type(n) is not int or not 1<=n<=100000 for g,n in floor_sizes.items())):
            raise ValueError('fixed complete FIT floor inventory required')
        self.floor_sizes=copy.deepcopy(floor_sizes)
        self.contract.update(schema='orbit-complete-demonstration-core-v1',floor_sizes=copy.deepcopy(floor_sizes),
            task_normalization='uniform-task minibatch estimator of equal-floor/equal-task/actual-action mean',
            endpoint='conditional teacher command+EOS token CE; actual recorded trajectory length',
            sampler='one fixed seeded permutation of all FIT tasks per phase',
            feedback='inner support only; no additional direct-rho observation loss',
            discrete_sampling_gradient=False)
        self.contract_identity=digest(self.contract)

    def _check(self):
        super()._check()
        if self.floor_sizes!=self.contract['floor_sizes']:
            raise ValueError('population normalization changed')

    def begin(self, groups, *, action_counts):
        expected={(g,v) for g,views in groups.items() for v in views}
        if (type(action_counts) is not dict or set(action_counts)!=expected
                or any(g not in self.floor_sizes for g in groups)
                or any(len(views)>self.floor_sizes[g] for g,views in groups.items())
                or any(type(n) is not int or not 1<=n<=64 for n in action_counts.values())):
            raise ValueError('complete original positive episode lengths and population required')
        super().begin(groups)
        self._batch['episodes']={key:dict(expected=n,actions=0,closed=False) for key,n in action_counts.items()}
        self._batch['tasks']=len(expected)

    def add_endpoint(self, *args, **kwargs):
        raise RuntimeError('use the complete-demonstration action accumulator')

    def add_action_batch(self, records, losses):
        self._check();batch=self._batch
        if (batch is None or type(records) not in (list,tuple) or not records
                or not torch.is_tensor(losses) or losses.ndim!=1 or len(losses)!=len(records)
                or not losses.requires_grad or not bool(torch.isfinite(losses).all())):
            raise ValueError('finite per-action losses from distinct current task states required')
        states=[];episodes=[];weights=[];seen=set()
        for group,task,index,fast in records:
            key=(group,task);episode=batch['episodes'].get(key)
            if (episode is None or episode['closed'] or key in seen
                    or type(index) is not int or index!=episode['actions'] or index>=episode['expected']):
                raise ValueError('one sequential observed action per live task')
            self._validate_fast(fast)
            tensors=(fast.a,fast.b,fast.momentum_a,fast.momentum_b,fast.rng_state,fast.initial_rng)
            storage={(str(p.device),p.untyped_storage().data_ptr()) for p in tensors}
            other={(str(p.device),p.untyped_storage().data_ptr()) for f in states for p in
                (f.a,f.b,f.momentum_a,f.momentum_b,f.rng_state,f.initial_rng)}
            if storage & other:
                raise ValueError('episodes must not share fast factors, optimizer or RNG')
            seen.add(key);states.append(fast);episodes.append(episode)
            weights.append(sum(self.floor_sizes.values())/(batch['tasks']*len(self.floor_sizes)*
                self.floor_sizes[group]*episode['expected']))
        fast_params=tuple(p for f in states for p in (f.a,f.b))
        targets=fast_params+tuple(p for _,p in self.named[2:])+(self.model.rho.a,self.model.rho.b)
        weighted=(losses*losses.new_tensor(weights)).sum()
        try:
            gradients=torch.autograd.grad(weighted,targets,allow_unused=True)
            n=len(fast_params)
            if any(g is None for g in gradients[:n]) or any(g is not None for g in gradients[-2:]):
                raise ValueError('task gradient must reach fast leaves, not directly reach rho')
            slow=(torch.stack(gradients[:n:2]).sum(0),torch.stack(gradients[1:n:2]).sum(0))+gradients[n:-2]
            self._accumulate(slow,1.)
        except BaseException:
            self.failed=True
            raise
        for episode in episodes:episode['actions']+=1
        batch['fast'].extend(states)
        batch['fast_storage'].update((str(p.device),p.untyped_storage().data_ptr()) for f in states for p in
            (f.a,f.b,f.momentum_a,f.momentum_b,f.rng_state,f.initial_rng))
        batch['endpoint_loss']+=float(weighted.detach())

    def close_episode(self,group,task):
        self._check();batch=self._batch
        if batch is None or (group,task) not in batch['episodes']:
            raise ValueError('registered complete demonstration required')
        episode=batch['episodes'][group,task]
        if episode['closed'] or episode['actions']!=episode['expected']:
            raise ValueError('all actual demonstration actions required before closing')
        episode['closed']=True;batch['seen'].add((group,task))

    def commit(self):
        self._check();batch=self._batch
        if batch is None or not all(e['closed'] for e in batch['episodes'].values()):
            raise ValueError('complete actual demonstrations required before slow update')
        episodes=copy.deepcopy(batch['episodes'])
        result=super().commit()
        result.update(actual_actions=sum(e['actions'] for e in episodes.values()),
            task_normalization=self.contract['task_normalization'],
            episodes=[dict(floor=g,task=t,**v) for (g,t),v in episodes.items()],
            model_task_success_claim=False)
        return result

    def restore_optimizer(self,saved):
        # Reuse only the audited generic AdamW state validator. No fixed-horizon
        # trajectory loss, begin/reset behavior or support objective is reused.
        from mmla_pdsa.full.outcome_trajectory import OutcomeTrajectoryCore
        return OutcomeTrajectoryCore.restore_optimizer(self,saved)
