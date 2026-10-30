"""Complete ordered logical blocks; proposal and audit decisions stay with the actor."""
from pathlib import Path
from dataclasses import asdict
from contextlib import closing
from production_common import read
from scalar_task_receipts import consume_job


class ScalarLogicalBackend:
    def __init__(self,pool,directory,*,source,support_receipts,instrument=False):
        self.pool=pool;self.directory=Path(directory);self.source=source
        self.support_receipts=support_receipts;self.completed_jobs=[];self.active=False;self.instrument=instrument

    def __call__(self,evaluator,rule,block):
        factory=evaluator.factory(rule);reader=factory.reader_identity()
        wire=dict(mode=rule.mode,symbols=list(rule.symbols),language=rule.language)
        requests=[dict(seed=evaluator.pair_seed,role=block.role,support_sha256=self.support_receipts[block.role],
            permit=permit.identity,index=index,rule=wire,instrument=self.instrument) for index,permit in enumerate(block.permits)]
        worlds=[];tokens=0;template=None;self.active=True
        per_world=64*factory.model.profile.max_new_tokens*factory.model.profile.local_tokens+63*4096
        with closing(self.pool.iter_ordered(requests)) as results:
            for ordinal,worker,response in results:
                request=requests[ordinal]
                result,reference=consume_job(self.directory,response,request,worker=worker,source=self.source,
                    terminal_kind='RTMA_TASK_TERMINAL',on_work=evaluator.on_work,on_event=evaluator.on_progress)
                evaluation=result['evaluation']
                if (evaluation['schema']!='orbit-fixed-phi-logical-block-v1' or evaluation['status']!='COMPLETE'
                        or evaluation['role']!=block.role or evaluation['block_sha256']!=block.block_sha256
                        or evaluation['population_sha256']!=block.population_sha256 or evaluation['reader']!=reader
                        or evaluation['rule']!=dict(mode=rule.mode,symbols=list(rule.symbols),language=rule.language)
                        or evaluation['control']!='candidate' or evaluation['pair_seed']!=evaluator.pair_seed
                        or evaluation['worlds']!=[result['endpoint']] or result['endpoint']['permit']!=request['permit']
                        or evaluation['forward_input_cap']!=per_world or not 0<evaluation['forward_input_tokens']<=per_world
                        or evaluation['admission_performed'] is not False or evaluation['successor_installed'] is not False):
                    raise ValueError('one exact admitted logical world required')
                worlds.append(result['endpoint']);tokens+=evaluation['forward_input_tokens'];template=evaluation
                self.completed_jobs.append(reference)
        if len(worlds)!=len(block.permits):raise ValueError('complete block required before feedback')
        self.active=False
        return dict(template,rule=asdict(rule),worlds=worlds,mean_loss=sum(row['loss'] for row in worlds)/len(worlds),
            forward_input_tokens=tokens,forward_input_cap=per_world*len(worlds),mean_cost=tokens/(per_world*len(worlds)))

    def boundary(self):
        if self.active or self.pool.failed or self.pool.closed or any(p.poll() is not None for p in self.pool.workers):
            raise RuntimeError('all logical jobs must be quiescent at the resource boundary')
        latest={row['worker']:row for row in self.completed_jobs}
        for worker,row in latest.items():
            value=read(self.directory/row['receipt'],expected=row['receipt_sha256'],cap=2*1024**2)
            if value['environment_closed'] is not True or value['GPU_synchronized'] is not True:
                raise ValueError('each worker must finish and synchronize its latest assigned job')
