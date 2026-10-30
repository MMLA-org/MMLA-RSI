"""Episode data parallelism with optional Megatron Core process groups.

This retains the exact Hugging Face mixer. It is NOT a Megatron GPT conversion,
Megatron DDP wrapper, tensor parallelism, or distributed optimizer. Fast state is
rank-local; only slow task-gradient sums and task counts cross ranks.
"""
from dataclasses import dataclass
from datetime import timedelta
import os
import torch
import torch.distributed as dist


@dataclass
class EpisodeParallel:
    group: object = None
    mode: str = 'single'
    owns_process_group: bool = False
    owns_megatron_state: bool = False

    def __post_init__(self):
        if self.mode=='single' and (int(os.environ.get('WORLD_SIZE','1'))!=1 or
                (dist.is_initialized() and dist.get_world_size()!=1)):
            raise ValueError('single mode cannot ignore launched ranks')

    @classmethod
    def initialize(cls, mode='torch', *, timeout_seconds=120):
        if mode not in ('single','torch','megatron-core') or not 1<=timeout_seconds<=300:
            raise ValueError('declared DP backend and bounded timeout required')
        if mode=='single':return cls()
        if mode=='torch' and not dist.is_initialized() and int(os.environ.get('WORLD_SIZE','1'))==1:return cls()
        owns=False;created=False;parallel_state=None
        try:
            if torch.cuda.is_available():
                # Object collectives also use this device with NCCL.
                torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            if not dist.is_initialized():
                if int(os.environ.get('WORLD_SIZE','1'))<2 and mode=='torch':return cls()
                dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo',
                    timeout=timedelta(seconds=timeout_seconds))
                owns=True
            if mode=='megatron-core':
                from megatron.core import parallel_state
                if parallel_state.model_parallel_is_initialized():
                    raise ValueError('externally initialized MCore topology is not supported')
                created=True
                parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,context_parallel_size=1,
                    expert_model_parallel_size=1,expert_tensor_parallel_size=1,
                    distributed_timeout_minutes=timeout_seconds/60,
                    gtp_remat_size=1,expert_gtp_remat_size=1,num_distributed_optimizer_instances=1)
                if (parallel_state.get_tensor_model_parallel_world_size()!=1
                        or parallel_state.get_pipeline_model_parallel_world_size()!=1
                        or parallel_state.get_context_parallel_world_size()!=1
                        or parallel_state.get_expert_model_parallel_world_size()!=1):
                    raise ValueError('this adapter supports pure data parallelism only')
                group=parallel_state.get_data_parallel_group(with_context_parallel=False,
                    with_gtp_remat=False,partial_data_parallel=False)
                wide=parallel_state.get_data_parallel_group(with_context_parallel=False,
                    with_gtp_remat=True,partial_data_parallel=False)
                if dist.get_world_size(wide)!=dist.get_world_size():
                    raise ValueError("undeclared GTP-remat topology")
                if dist.get_world_size(group)!=dist.get_world_size():
                    raise ValueError('DP group must cover every launched rank')
                return cls(group,mode,owns,created)
            return cls(dist.group.WORLD,mode,owns)
        except Exception:
            if created and parallel_state is not None:parallel_state.destroy_model_parallel()
            if owns and dist.is_initialized():dist.destroy_process_group()
            raise

    @property
    def size(self):return dist.get_world_size(self.group) if self.mode!='single' else 1

    @property
    def rank(self):return dist.get_rank(self.group) if self.mode!='single' else 0

    def agree(self, success, *, device):
        # With one participant there is no device collective to synchronize.
        if self.size==1:return bool(success)
        value=torch.tensor(int(bool(success)),dtype=torch.int32,device=device)
        if self.size>1:dist.all_reduce(value,op=dist.ReduceOp.MIN,group=self.group)
        return bool(value.item())

    def verify_parameters(self, parameters):
        """Verify common slow tensors; per-process rho UUIDs are not compared."""
        from .neural import tensor_digest
        identity=tensor_digest({str(i):p for i,p in enumerate(parameters)})
        if self.size>1:
            hashes=[None]*self.size
            dist.all_gather_object(hashes,identity,group=self.group)
            if len(set(hashes))!=1:raise ValueError('slow state differs across ranks')
        return identity

    def gather(self, value):
        if self.size==1:return [value]
        values=[None]*self.size
        dist.all_gather_object(values,value,group=self.group)
        return values

    def verify_batch(self, groups, contract):
        """No duplicated semantic task, including padded distributed tails."""
        rows=self.gather((tuple(groups), contract))
        all_groups=[g for groups,_ in rows for g in groups]
        if not all_groups or len(set(all_groups))!=len(all_groups):
            raise ValueError('global outer batch repeats or lacks a semantic group')
        if any(c!=rows[0][1] for _,c in rows):
            raise ValueError('outer objective/optimizer/configuration differs across ranks')

    def reduce_task_sums(self, parameters, *, local_tasks):
        if type(local_tasks) is not int or local_tasks<0 or not parameters:
            raise ValueError('nonnegative real local task count required')
        device=parameters[0].device
        safe=all(not hasattr(p,'main_grad') for p in parameters)
        if not self.agree(safe,device=device):raise ValueError('MCore DDP main_grad is not this adapter')
        if self.size==1:total=local_tasks
        else:
            count=torch.tensor(local_tasks,dtype=torch.int64,device=device)
            dist.all_reduce(count,group=self.group)
            total=int(count.item())
        if total<=0:raise ValueError('empty global outer batch')
        # Same ordered parameter list on every rank, including empty ranks.
        # One collective per dtype/device bucket replaces tiny per-parameter
        # calls. Missing gradients remain zeros; normalization uses real tasks.
        groups={}
        for p in parameters:
            groups.setdefault((p.device,p.dtype),[]).append(p)
        for members in groups.values():
            bucket=[];elements=0
            def reduce_bucket(rows):
                flat=torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().reshape(-1) for p in rows])
                if self.size>1:dist.all_reduce(flat,group=self.group)
                flat.div_(total);offset=0
                for p in rows:
                    p.grad=flat[offset:offset+p.numel()].view_as(p);offset+=p.numel()
            for p in members:
                if bucket and elements+p.numel()>4*1024*1024:
                    reduce_bucket(bucket);bucket=[];elements=0
                bucket.append(p);elements+=p.numel()
            if bucket:reduce_bucket(bucket)
        return total

    def shard(self, ordered_tasks):
        return tuple(ordered_tasks[self.rank::self.size])

    def close(self):
        if self.owns_megatron_state:
            from megatron.core import parallel_state
            parallel_state.destroy_model_parallel()
        if self.owns_process_group and dist.is_initialized():dist.destroy_process_group()
