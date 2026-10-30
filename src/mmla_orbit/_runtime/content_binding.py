"""Ten-feature proposal factory using the original frozen model interfaces."""
from dataclasses import asdict
import json
from mmla_pdsa.full.profile import digest
from mmla_pdsa.full.neural import tensor_digest
from content_rules import OPS,parse_rule
from content_proposals import TokenCodec,ModelProposer
from content_roles import ProposalLearner
from content_evidence import EvidenceMemory

def real_token_codec(tokenizer,language,mode,vocab):
    """Explicit code symbols A..; every symbol is one full, distinct model token."""
    symbols=tuple('c'+str(i) for i in range(len(language.constants)))+tuple('x'+str(i) for i in range(8))+OPS+('x8','x9')
    if len(symbols)>52:raise ValueError('bounded printable token codebook')
    alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    mapping={}
    for symbol,text in zip(symbols,alphabet):
        ids=tokenizer.encode(text,add_special_tokens=False)
        if len(ids)!=1 or tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=text:
            raise ValueError('codebook character must be exactly one roundtrip model token')
        mapping[symbol]=ids[0]
    required=set(symbols)-({f'x{i}' for i in range(4,10)} if mode=='rtt' else set())
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
        return digest(dict(schema='rsi-content-proposal-factory-v1',actual_model=self.actual_model_configuration(),header_schema='complete-rsi-content-interface-v1',model=self.model_identity,
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
          '\nContent V features x8=newly retained coverage, x9=lost retained coverage; NULL has zero of both. Codebook symbol to literal token: '+json.dumps(alphabet,sort_keys=True))
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
