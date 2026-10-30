"""Production role ownership, real-token codebook and immutable role packages."""
from dataclasses import asdict
from pathlib import Path
import copy,json,os,shutil,tempfile
import torch
from ..neural import LearningResidual,tensor_digest
from ..profile import wire,digest
from ..checkpoint import save_bundle,load_bundle,_sha
from .rules import OPS,parse_rule
from .proposals import TokenCodec,ModelProposer
from .roles import ProposalLearner
from .evidence import EvidenceMemory


def rebind(model):
    if hasattr(model,'hybrid'):model._bind_configuration_buffer()
    if hasattr(model,'joint_risk'):model._bind_joint_schema()
    model.rho.invalidate()
    return model


def isolated_model(prototype):
    """Copy new modules only; immutable theta0/omega is the shared substrate."""
    return rebind(copy.deepcopy(prototype,{id(prototype.mixer):prototype.mixer}))


def initialize_proposal_role(task_model,*,seed):
    if any(m.training for m in task_model.modules()):raise ValueError('freeze task model before role fork')
    memo={id(m):m for k,m in task_model._modules.items() if k!='rho'}
    proposal=copy.deepcopy(task_model,memo)
    proposal.rho=LearningResidual(task_model.profile.hidden_size,seed=seed).to(device=task_model.rho.a.device,dtype=torch.float32).eval()
    proposal.rho.initialization=dict(a=proposal.rho.a.detach().clone(),b=proposal.rho.b.detach().clone(),seed=seed)
    proposal.module_status=dict(task_model.module_status,rho='INITIALIZED_PROPOSAL_ROLE_NOT_TRAINED')
    rebind(proposal);check_role_pair(task_model,proposal)
    return proposal


def check_role_pair(task,proposal):
    if (type(task) is not type(proposal) or task is proposal or task.profile!=proposal.profile
            or task.substrate_identity!=proposal.substrate_identity or set(task._modules)!=set(proposal._modules)
            or task.rho is proposal.rho or tuple(task.rho.a.shape)!=tuple(proposal.rho.a.shape)
            or tuple(task.rho.b.shape)!=tuple(proposal.rho.b.shape)):
        raise ValueError('same layout and substrate; distinct role initializations')
    for k,m in task._modules.items():
        if k!='rho' and (proposal._modules[k] is not m or any(p.requires_grad for p in m.parameters())):
            raise ValueError('only frozen common modules may be shared')
    for a,b in [(task.rho.a,proposal.rho.a),(task.rho.b,proposal.rho.b),(task.rho.version,proposal.rho.version)]:
        if a.data_ptr()==b.data_ptr():raise ValueError('role rho storage aliases')
        if a.dtype!=b.dtype or a.device!=b.device:raise ValueError('same role dtype and device layout')
    if any(x.dtype!=torch.float32 for m in (task,proposal) for x in (m.rho.a,m.rho.b)):
        raise ValueError('role residual factors must be FP32')
    if task.rho.identity==proposal.rho.identity:raise ValueError('separate initialization ownership required')


def real_token_codec(tokenizer,language,mode,vocab):
    """Explicit code symbols A..; every symbol is one full, distinct model token."""
    symbols=tuple('c'+str(i) for i in range(len(language.constants)))+tuple('x'+str(i) for i in range(8))+OPS
    if len(symbols)>52:raise ValueError('bounded printable token codebook')
    alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    mapping={}
    for symbol,text in zip(symbols,alphabet):
        ids=tokenizer.encode(text,add_special_tokens=False)
        if len(ids)!=1 or tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=text:
            raise ValueError('codebook character must be exactly one roundtrip model token')
        mapping[symbol]=ids[0]
    required=set(symbols)-({f'x{i}' for i in range(4,8)} if mode=='rtt' else set())
    codec=TokenCodec(tuple((s,mapping[s]) for s in symbols if s in required))
    codec.validate(language,mode,vocab)
    return codec


class ProductionProposalFactory:
    """Every generation starts fresh Phi/M; the accepted parent owns adaptation."""
    def __init__(self,model,tokenizer,*,tokenizer_identity,language,memory,attempts,update_bounds,
                 temperature=1.,replay_tolerance=1e-4):
        if (memory.reader_identity!=tokenizer_identity or memory.version or any(memory.rows)
                or type(attempts) is not int or not 2<=attempts<=1024):raise ValueError('empty bound proposal memory and finite opportunities')
        self.model,self.tokenizer,self.tokenizer_identity=model,tokenizer,tokenizer_identity
        self.language,self.memory,self.attempts,self.bounds=language,memory,attempts,update_bounds
        self.temperature,self.tolerance=temperature,replay_tolerance
        self.codecs={mode:real_token_codec(tokenizer,language,mode,model.mixer.get_input_embeddings().weight.shape[0])
                     for mode in ('rtt','rtma','joint')}
        self._guard=self.guard();self._tokenizer=tokenizer
        self.model_identity=digest(dict(profile=asdict(model.profile),substrate=model.substrate_identity,
            parameters=tensor_digest(model.state_dict()),initialization=tensor_digest({k:model.rho.initialization[k] for k in ('a','b')})))
        self.identity=self.configuration()
    def guard(self):
        return tuple((k,id(x),x._version,x.data_ptr(),str(x.device),str(x.dtype),tuple(x.shape))
            for k,x in list(self.model.named_parameters())+list(self.model.named_buffers()))
    def actual_model_configuration(self):
        return dict(profile=asdict(self.model.profile),substrate=self.model.substrate_identity,
            mixer=self.model.mixer.config.to_dict(),
            hybrid=None if not hasattr(self.model,'hybrid_config') else asdict(self.model.hybrid_config))
    def configuration(self):
        return digest(dict(schema='rsi-production-proposal-factory-v1',actual_model=self.actual_model_configuration(),header_schema='complete-rsi-interface-v2',model=self.model_identity,
            tokenizer=self.tokenizer_identity,language=self.language.identity,memory0=self.memory.identity,
            codecs={k:v.identity for k,v in self.codecs.items()},attempts=self.attempts,
            bounds=asdict(self.bounds),temperature=self.temperature,replay_tolerance=self.tolerance))
    def header(self,parent):
        codec=self.codecs[parent.mode]
        alphabet={s:self.tokenizer.decode([i],skip_special_tokens=False) for s,i in codec.pairs}
        return ('Propose a bounded adaptation program in prefix notation. Emit only code tokens. '
          'Binary operators add, sub, mul, min, max, sdiv; protected division uses the fixed language epsilon. '
          'RTT has three trees for step size, clipping, momentum; RTMA has one tree scoring NULL or row replacement. '
          'Joint concatenates RTT then RTMA. Current parent: '+json.dumps(asdict(parent),sort_keys=True)+
          '\nLanguage: '+json.dumps(asdict(self.language),sort_keys=True)+
          '\nConstants c_i: '+json.dumps([self.language.constant(i) for i in range(len(self.language.constants))])+
          '\nU features x0=loss, x1=clip(loss-previous_loss,-1,1), x2=min(1,gradient_L2/gradient_reference), x3=opportunity/opportunity_cap. '+
          'V features x0=is_NULL, x1=slot_occupied, x2=new_loss, x3=new_cost, x4=old_loss, x5=old_cost, x6=min(1,age/opportunity_cap), x7=occupied_fraction; old values and age are zero for empty slots. '+
          'V chooses maximum score, NULL first then ascending slot on ties. '+
          'U clamps outputs to bounds, uses mu*momentum+(1-mu)*clipped_gradient, clips momentum, and projects factors around role rho. '+
          'Bounds: '+json.dumps(asdict(self.bounds),sort_keys=True)+
          '\nCodebook symbol to literal token: '+json.dumps(alphabet,sort_keys=True))
    def create(self,parent,generation,seed,on_forward):
        if parent.language!=self.language.identity or parse_rule(self.language,parent.mode,parent.symbols)!=parent:
            raise ValueError('parent must be the canonical rule in this exact language')
        if (self.guard()!=self._guard or self.tokenizer is not self._tokenizer or self.configuration()!=self.identity
                or any(m.training for m in self.model.modules()) or type(generation) is not int or generation<0):
            raise ValueError('frozen exact proposal role configuration')
        memory=EvidenceMemory.restore(self.memory.export(),count_tokens=self.memory._count_tokens,expected_contract=self.memory.contract)
        proposer=ModelProposer(self.model,self.tokenizer,language=self.language,mode=parent.mode,
            codec=self.codecs[parent.mode],header=self.header(parent),parent_identity=parent.identity,
            memory_token_allowance=memory.bounds.read_tokens,tokenizer_identity=self.tokenizer_identity,
            temperature=self.temperature,replay_absolute_tolerance=self.tolerance,on_forward=on_forward)
        return ProposalLearner(proposer,parent,memory,seed=seed,opportunity_cap=self.attempts,update_bounds=self.bounds)


def save_role_pair(task,proposal,destination,*,contract,task_trainer=None,proposal_trainer=None):
    check_role_pair(task,proposal)
    from ..training import SlowTrainer
    if any(t is not None and type(t) is not SlowTrainer for t in (task_trainer,proposal_trainer)):
        raise ValueError('paired role packages own only two separate rho optimizers')
    if not contract:raise ValueError('explicit complete production role contract')
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError(dest)
    tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        save_bundle(task,tmp/'task',trainer=task_trainer)
        save_bundle(proposal,tmp/'proposal',trainer=proposal_trainer)
        metadata=dict(schema='mmla-rsi-role-pair-v1',contract=contract,profile=task.profile.identity,
            substrate=task.substrate_identity,
            shared_modules=tensor_digest({k:v for k,v in task.new_state_dict().items() if not k.startswith('rho.')}),
            roles={role:dict(manifest_sha256=_sha(tmp/role/'manifest.json'),
                rho=tensor_digest(model.rho.state_dict()),initialization_seed=model.rho.initialization['seed'],
                status=model.module_status['rho'],optimizer_saved=trainer is not None)
                for role,model,trainer in [('task',task,task_trainer),('proposal',proposal,proposal_trainer)]})
        (tmp/'roles.json').write_bytes(wire(metadata)+b'\n')
        with (tmp/'roles.json').open('rb') as f:os.fsync(f.fileno())
        fd=os.open(tmp,os.O_RDONLY);os.fsync(fd);os.close(fd)
        if dest.exists():raise FileExistsError(dest)
        os.rename(tmp,dest);fd=os.open(dest.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
        return metadata
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise


def load_role_pair(prototype,directory,*,contract,trainer_factories=None):
    """Validate isolated models before returning them; caller's prototype stays intact."""
    root=Path(directory);meta=json.loads((root/'roles.json').read_bytes())
    if (meta['schema']!='mmla-rsi-role-pair-v1' or wire(meta['contract'])!=wire(contract)
            or meta['profile']!=prototype.profile.identity or meta['substrate']!=prototype.substrate_identity):
        raise ValueError('exact declared paired role package required')
    factories=trainer_factories or {};models={};trainers={};restored={}
    for name in ('task','proposal'):
        row=meta['roles'][name]
        if _sha(root/name/'manifest.json')!=row['manifest_sha256']:raise ValueError('role child manifest changed')
        model=isolated_model(prototype);trainer=factories[name](model) if name in factories else None
        from ..training import SlowTrainer
        if trainer is not None and type(trainer) is not SlowTrainer:raise ValueError('role optimizer must own only rho')
        if row['optimizer_saved']!=(trainer is not None):raise ValueError('restore every saved optimizer explicitly')
        restored[name]=load_bundle(model,root/name,trainer=trainer)
        if (tensor_digest(model.rho.state_dict())!=row['rho'] or model.rho.initialization['seed']!=row['initialization_seed']
                or model.module_status['rho']!=row['status']
                or tensor_digest({k:v for k,v in model.new_state_dict().items() if not k.startswith('rho.')})!=meta['shared_modules']):
            raise ValueError('role or common module content mismatch')
        for key,module in model._modules.items():
            if key!='rho':module.requires_grad_(False)
        model.eval()
        models[name]=model;trainers[name]=trainer
    task,proposal=models['task'],models['proposal']
    # Equality was checked before reusing the common immutable modules.
    for k,m in task._modules.items():
        if k!='rho':proposal._modules[k]=m
    rebind(proposal);check_role_pair(task,proposal)
    return dict(models=models,trainers=trainers,recovery=restored,manifest=meta)
