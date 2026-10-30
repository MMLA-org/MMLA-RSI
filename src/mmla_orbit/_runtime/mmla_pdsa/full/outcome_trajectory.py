"""Streaming complete-task objective over separately timed fast states.

Each actual action has a teacher-label event (or an explicit unavailable label)
and one actual observation-feedback support event. All tasks retain their fixed
horizon denominator, including early success and teacher failure. This is a
conditional imitation/modeling surrogate; final task success is scored separately.
"""
import copy
import math
import torch
from .outcome_training import OutcomeUpdateCore
from .profile import digest
from .module_training import finite_optimizer


class OutcomeTrajectoryCore(OutcomeUpdateCore):
    def __init__(self, model, config, *, horizon, support_weight):
        super().__init__(model, config)
        if (type(horizon) is not int or not 1 <= horizon <= 512
                or type(support_weight) not in (int, float) or not math.isfinite(support_weight)
                or not 0 < support_weight <= 10):
            raise ValueError('fixed finite task horizon and support mixture required')
        self.horizon, self.support_weight = horizon, support_weight
        self.contract.update(schema='orbit-task-trajectory-core-v1', horizon=horizon,
            support_weight=support_weight, task_normalization='equal-floor/equal-task/fixed-horizon',
            teacher_missing='explicit-unavailable; zero-gradient; task-retained',
            task_endpoint='conditional-next-action-CE; not-differentiated-won')
        self.contract_identity = digest(self.contract)

    def _check(self):
        super()._check()
        if self.contract['horizon'] != self.horizon or self.contract['support_weight'] != self.support_weight:
            raise RuntimeError('trajectory settings changed')

    def begin(self, groups):
        super().begin(groups)
        self._batch['trajectories'] = {(g,v): dict(events=[],supports=0,closed=False,terminal=None)
                                      for g,views in groups.items() for v in views}
        self._batch['support_loss'] = 0.

    def _episode(self, group, view, step):
        self._check()
        if self._batch is None or (group, view) not in self._batch['trajectories']:
            raise ValueError('registered complete task required')
        episode = self._batch['trajectories'][group, view]
        if episode['closed'] or type(step) is not int or not 0 <= step < self.horizon:
            raise ValueError('live bounded task opportunity required')
        return episode

    def add_task_loss(self, group, view, step, fast, loss):
        episode = self._episode(group, view, step)
        if step != len(episode['events']) or step != episode['supports']:
            raise ValueError('one current-state teacher event before each actual action')
        # Base core checks FO-rho/direct-hybrid ownership on this exact state's
        # independent leaf factors. A later state contributes a new gradient;
        # different-time losses are never attached to a single final Phi.
        super().add_endpoint(group, view, fast, loss/self.horizon)
        self._batch['seen'].remove((group, view))
        episode['events'].append('SUPERVISED')

    def add_task_batch(self, records, losses):
        """One backward for independent rows, then FO-transfer each fast gradient.

        records contains (group, task, action_index, exact endpoint_fast).
        No padded or missing-label task is duplicated to fill the GPU batch.
        """
        self._check()
        if (type(records) not in (tuple,list) or not records or losses.ndim!=1
                or len(losses)!=len(records) or not losses.requires_grad or not torch.isfinite(losses).all()):
            raise ValueError('complete independently scored endpoint rows required')
        episodes=[];weights=[];states=[];seen=set()
        for group,view,step,fast in records:
            episode=self._episode(group,view,step)
            if (step!=len(episode['events']) or step!=episode['supports'] or (group,view) in seen):
                raise ValueError('one current state per distinct task in endpoint batch')
            self._validate_fast(fast)
            tensors=(fast.a,fast.b,fast.momentum_a,fast.momentum_b,fast.rng_state,fast.initial_rng)
            if any(p.untyped_storage().data_ptr()==q.untyped_storage().data_ptr()
                    for old in states for p in tensors for q in (old.a,old.b,old.momentum_a,old.momentum_b,old.rng_state,old.initial_rng)):
                raise ValueError('endpoint rows share fast factors, optimizer or RNG')
            seen.add((group,view));episodes.append(episode);states.append(fast)
            weights.append(1/len(self._batch['layout'])/len(self._batch['layout'][group])/self.horizon)
        fast_params=tuple(p for fast in states for p in (fast.a,fast.b))
        targets=fast_params+tuple(p for _,p in self.named[2:])+(self.model.rho.a,self.model.rho.b)
        try:
            gs=torch.autograd.grad((losses*losses.new_tensor(weights)).sum(),targets,allow_unused=True)
            n=len(fast_params)
            if any(x is None for x in gs[:n]) or gs[-2] is not None or gs[-1] is not None:
                raise ValueError('endpoint gradients must reach every fast row and not slow rho directly')
            gradients=(torch.stack(gs[:n:2]).sum(0),torch.stack(gs[1:n:2]).sum(0))+gs[n:-2]
            self._accumulate(gradients,1.)
        except BaseException:
            self.failed=True
            raise
        for episode in episodes:episode['events'].append('SUPERVISED')
        self._batch['fast'].extend(states)
        self._batch['fast_storage'].update((str(p.device),p.untyped_storage().data_ptr()) for fast in states for p in
            (fast.a,fast.b,fast.momentum_a,fast.momentum_b,fast.rng_state,fast.initial_rng))
        self._batch['endpoint_loss']+=float((losses.detach()*losses.new_tensor(weights)).sum())

    def add_support_batch(self, records, losses):
        """One direct-rho backward for actual feedback rows on their old contexts."""
        self._check()
        if (type(records) not in (tuple,list) or not records or losses.ndim!=1
                or len(losses)!=len(records) or not losses.requires_grad or not torch.isfinite(losses).all()):
            raise ValueError('complete independently scored support rows required')
        episodes=[];weights=[];seen=set()
        for group,view,step in records:
            episode=self._episode(group,view,step)
            if step!=episode['supports'] or len(episode['events'])!=step+1 or (group,view) in seen:
                raise ValueError('one actual feedback per distinct task row')
            seen.add((group,view));episodes.append(episode)
            weights.append(self.support_weight/len(self._batch['layout'])/len(self._batch['layout'][group])/self.horizon)
        loss=(losses*losses.new_tensor(weights)).sum()
        try:
            gs=torch.autograd.grad(loss,tuple(p for _,p in self.named),allow_unused=True)
            if gs[0] is None or gs[1] is None:raise ValueError('support uses slow rho directly')
            self._accumulate(gs,1.)
        except BaseException:
            self.failed=True
            raise
        for episode in episodes:episode['supports']+=1
        self._batch['support_loss']+=float(loss.detach())

    def no_teacher(self, group, view, step, *, reason):
        episode = self._episode(group, view, step)
        if (step != len(episode['events']) or step != episode['supports']
                or reason not in ('NO_ADMISSIBLE_LABEL','HandCodedAgentFailed','HandCodedAgentTimeout')):
            raise ValueError('explicit teacher outcome at the current state required')
        episode['events'].append(reason)

    def add_support_loss(self, group, view, step, loss):
        episode = self._episode(group, view, step)
        if step != episode['supports'] or len(episode['events']) != step+1:
            raise ValueError('one actual feedback support loss after its action event')
        self._loss(loss)
        weight = self.support_weight/len(self._batch['layout'])/len(self._batch['layout'][group])/self.horizon
        try:
            gradients = torch.autograd.grad(loss, tuple(p for _,p in self.named), allow_unused=True)
            if gradients[0] is None or gradients[1] is None:
                raise ValueError('outer support objective must use slow rho directly')
            self._accumulate(gradients, weight)
        except BaseException:
            self.failed = True
            raise
        episode['supports'] += 1
        self._batch['support_loss'] += float(loss.detach())*weight

    def close_episode(self, group, view, *, actions, termination):
        self._check()
        if self._batch is None or (group,view) not in self._batch['trajectories']:
            raise ValueError('registered task closure required')
        episode = self._batch['trajectories'][group,view]
        if (episode['closed'] or type(actions) is not int or not 0 <= actions <= self.horizon
                or actions != len(episode['events']) or actions != episode['supports']
                or termination not in ('WON','LOST','STEP_LIMIT')
                or (termination == 'STEP_LIMIT' and actions != self.horizon)):
            raise ValueError('complete task termination and every actual event required')
        episode['closed'] = True;episode['terminal'] = termination
        self._batch['seen'].add((group,view))

    def commit(self):
        self._check()
        if self._batch is None or not all(x['closed'] for x in self._batch['trajectories'].values()):
            raise ValueError('all planned tasks must close before slow update')
        traces = copy.deepcopy(self._batch['trajectories'])
        support = self._batch['support_loss']
        result = super().commit()
        result.update(support_loss=support, endpoint_horizon=self.horizon,
            teacher_events=sum(len(x['events']) for x in traces.values()),
            teacher_unavailable=sum(e!='SUPERVISED' for x in traces.values() for e in x['events']),
            episodes=[dict(group=g,view=v,**x) for (g,v),x in traces.items()])
        return result

    def restore_optimizer(self, saved):
        """Restore only after the full model has been restored and verified.

        The outer checkpoint must also bind its schedule cursor, RNG, resource
        ledger and source manifests. This method cannot authorize a new run or
        replay interrupted work and never replaces original model parameters.
        """
        self._check()
        if self._batch is not None or self.steps or self.optimizer.state:
            raise ValueError('restore into fresh idle core only')
        if (set(saved)!={'schema','contract','steps','rho_version','parameters','optimizer'}
                or saved['schema']!=self.contract['schema'] or saved['contract']!=self.contract
                or type(saved['steps']) is not int or not 0<=saved['steps']<=self.config.max_steps
                or type(saved['rho_version']) is not int or saved['rho_version']!=int(self.model.rho.version)
                or set(saved['parameters'])!={n for n,p in self.named}):
            raise ValueError('exact training contract and completed state required')
        for name,p in self.named:
            value=saved['parameters'][name]
            if (not torch.is_tensor(value) or value.shape!=p.shape or value.dtype!=p.dtype
                    or not torch.equal(value.to(p.device),p.detach())):
                raise ValueError('full restored model differs from optimizer checkpoint')
        try:
            self.optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
            finite_optimizer(self.optimizer)
            for name,p in self.named:
                state=self.optimizer.state.get(p)
                if saved['steps']==0:
                    if state:raise ValueError('unstepped optimizer cannot carry moments')
                elif (not state or set(state)!={'step','exp_avg','exp_avg_sq'}
                        or float(state['step'])!=saved['steps']
                        or any(state[k].shape!=p.shape or state[k].dtype!=p.dtype for k in ('exp_avg','exp_avg_sq'))):
                    raise ValueError('complete AdamW moments and counters required')
            self.steps=saved['steps'];self._check()
        except BaseException:
            self.failed=True
            raise
