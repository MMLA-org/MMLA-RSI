"""Actual grammar-masked proposal sampling and bounded score-function replay.

The fixed header and complete current memory read are the entire input. No
archive or prior input KV survives an attempt. Replay targets the sampled
candidate's development loss, not the whole recursive trajectory objective.
"""
from dataclasses import dataclass, asdict, replace
import math
import time
import torch
from .neural import tensor_digest
from .rules import PrefixGrammar, identity
from .reader_binding import FrozenReaderBinding


@dataclass(frozen=True)
class TokenCodec:
    pairs: tuple[tuple[str, int], ...]

    def validate(self, language, mode, vocab):
        mapping = dict(self.pairs)
        required = {'c'+str(i) for i in range(len(language.constants))}
        required |= {'x'+str(i) for i in range(4 if mode == 'rtt' else 8)}
        from .rules import OPS
        required |= set(OPS)
        if (set(mapping) != required or len(mapping) != len(self.pairs)
                or len(set(mapping.values())) != len(mapping)
                or any(type(v) is not int or not 0 <= v < vocab for v in mapping.values())):
            raise ValueError('one unique actual model token per declared grammar symbol')
        return mapping

    @property
    def identity(self):
        return identity(self.pairs)


@dataclass(frozen=True)
class Rollout:
    tokens: tuple[int, ...]
    symbols: tuple[str, ...]
    masks: tuple[tuple[int, ...], ...]
    prompt: tuple[int, ...]
    sampled_log_probability: float
    rule: object
    memory_read: dict
    memory_text: str
    policy_read: str
    rng_before: torch.Tensor
    rng_after: torch.Tensor
    parent_identity: str
    sampling_identity: str
    failure: str | None
    cost: dict
    seal: str = ''

    def digest(self):
        data = asdict(self)
        data.pop('seal')
        for key in ('rng_before', 'rng_after'):
            data[key] = tensor_digest({key:getattr(self,key)})
        return identity(data)


class SamplingFailure(RuntimeError):
    """Operational failure stops the caller, retaining the actual partial draw."""
    def __init__(self, rollout):
        super().__init__(rollout.failure)
        self.rollout = rollout


class ModelProposer:
    def __init__(self, model, tokenizer, *, language, mode, codec, header,
                 parent_identity, memory_token_allowance, tokenizer_identity,
                 temperature=1., replay_absolute_tolerance=1e-4, on_forward=None,
                 reader_binding=None):
        if type(header) is not str or not header or not parent_identity or not tokenizer_identity:
            raise ValueError('fixed parent, tokenizer and interface header required')
        if (not 0 < temperature < math.inf or not 0 <= replay_absolute_tolerance < math.inf
                or type(memory_token_allowance) is not int or memory_token_allowance < 1):
            raise ValueError('finite bound sampling/replay configuration')
        self.model, self.tokenizer = model, tokenizer
        self._bound_tokenizer = tokenizer
        self.tokenizer_identity = tokenizer_identity
        self.language, self.mode, self.codec = language, mode, codec
        self.mapping = codec.validate(language, mode, model.mixer.get_input_embeddings().weight.shape[0])
        self.header, self.parent_identity = header, parent_identity
        self.temperature, self.on_forward = temperature, on_forward
        self.memory_token_allowance = memory_token_allowance
        self.replay_absolute_tolerance = replay_absolute_tolerance
        # Reuse an actual resident-model binding, or hash once for a standalone
        # learner. Tensor/role guards reject supported writes; unsafe .data and
        # raw-storage writes are outside the original reader's supported API.
        self._bound_guard = self._guard()
        if reader_binding is None:
            reader_binding = FrozenReaderBinding(model, tokenizer, tokenizer_identity=tokenizer_identity)
        if type(reader_binding) is not FrozenReaderBinding:
            raise ValueError('actual process-local frozen reader binding required')
        self._reader_binding = reader_binding
        self.reader_identity = reader_binding.validate(model, tokenizer, tokenizer_identity=tokenizer_identity)
        self.sampling_identity = self._configuration_identity()
        if (len(tokenizer.encode(header, add_special_tokens=False)) + memory_token_allowance
                + language.max_tokens + 16 > model.profile.local_tokens):
            raise ValueError('header + declared memory + rollout cannot fit context')

    def _guard(self):
        return tuple((name,id(x),x._version,str(x.device),str(x.dtype),tuple(x.shape),x.data_ptr())
                     for name,x in list(self.model.named_parameters())+list(self.model.named_buffers()))

    def _configuration_identity(self):
        return identity(dict(header=self.header, codec=self.codec.identity,
            language=self.language.identity, mode=self.mode, reader=self.reader_identity,
            tokenizer=self.tokenizer_identity, parent=self.parent_identity,
            temperature=self.temperature, memory_allowance=self.memory_token_allowance,
            replay_absolute_tolerance=self.replay_absolute_tolerance))

    def _check_reader(self):
        self._reader_binding.validate(self.model, self.tokenizer, tokenizer_identity=self.tokenizer_identity)
        if (self.model.training or self._guard() != self._bound_guard
                or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')
                or self.tokenizer is not self._bound_tokenizer
                or self._configuration_identity() != self.sampling_identity
                or self.mapping != self.codec.validate(self.language, self.mode,
                    self.model.mixer.get_input_embeddings().weight.shape[0])):
            raise ValueError('bound reader/eval/sampling/numerical configuration changed')

    def _prompt(self, text):
        return tuple(self.tokenizer.encode(self.header+'\nMEMORY\n'+text+'\nPROGRAM\n', add_special_tokens=False))

    def _forward(self, ids, fast, positions):
        if self.on_forward:
            self.on_forward(len(ids), len(positions))
        device = self.model.rho.a.device
        ids = torch.tensor([ids], device=device, dtype=torch.long)
        return self.model.forward_batch(ids, torch.ones_like(ids), fast_states=[fast],
            memory=None, logit_positions=torch.tensor(positions, device=device, dtype=torch.long))[0]

    def sample(self, fast, memory):
        self._check_reader()
        self.model.rho.factors(fast)
        text, read = memory.read()
        if read['read_tokens'] > self.memory_token_allowance:
            raise ValueError('memory read exceeds prebound proposal allowance')
        prompt = self._prompt(text)
        if not prompt or len(prompt)+self.language.max_tokens > self.model.profile.local_tokens:
            raise ValueError('complete joined input exceeds context; no truncated rows')
        before = fast.rng_state.clone()
        generator = torch.Generator(device='cpu');generator.set_state(before)
        grammar = PrefixGrammar(self.language, self.mode)
        tokens, masks = [], []
        total, calls, input_tokens = 0., 0, 0
        start = time.monotonic()
        operational_error = None
        try:
            with torch.no_grad():
                while not grammar.complete and len(tokens) < self.language.max_tokens:
                    allowed = grammar.allowed()
                    if not allowed:
                        break
                    permitted = tuple(self.mapping[s] for s in allowed)
                    current = prompt+tuple(tokens)
                    calls += 1;input_tokens += len(current)
                    logits = self._forward(current, fast, [len(current)-1])[0]
                    scores = logits[list(permitted)].float().cpu()/self.temperature
                    if not torch.isfinite(scores).all():
                        raise FloatingPointError('nonfinite proposal logits')
                    logp = scores.log_softmax(0)
                    selected = int(torch.multinomial(logp.exp(), 1, generator=generator))
                    tokens.append(permitted[selected]);masks.append(permitted)
                    total += float(logp[selected]);grammar.append(allowed[selected])
        except Exception as exc:
            operational_error = exc
        finally:
            # Randomness already spent is retained even if the next update fails.
            fast.rng_state = generator.get_state()
        failure = None
        rule = None
        if operational_error is not None:
            failure = 'operational:'+type(operational_error).__name__
        elif not grammar.complete:
            failure = 'incomplete-prefix'
        else:
            try:
                rule = grammar.finish()
            except ValueError:
                failure = 'serialization-capacity'
        result = Rollout(tuple(tokens), tuple(grammar.symbols), tuple(masks), prompt, total,
            rule, read, text, fast.identity, before, fast.rng_state.clone(), self.parent_identity,
            self.sampling_identity, failure,
            dict(forward_calls_attempted=calls, input_tokens_attempted=input_tokens,
                 wall_seconds=time.monotonic()-start))
        result = replace(result,seal=result.digest())
        if operational_error is not None:
            raise SamplingFailure(result) from operational_error
        return result

    def gradient(self, fast, rollout, arrived_loss, *, on_backward=lambda n:None):
        self._check_reader()
        if (not 0 <= arrived_loss <= 1 or fast.identity != rollout.policy_read
                or self.parent_identity != rollout.parent_identity
                or rollout.sampling_identity != self.sampling_identity
                or rollout.digest() != rollout.seal):
            raise ValueError('feedback or rollout does not bind the actual sampler/reader')
        if (rollout.failure or '').startswith('operational:'):
            raise ValueError('operational sampling failure terminates the run')
        if (rollout.prompt != self._prompt(rollout.memory_text)
                or not len(rollout.tokens) == len(rollout.symbols) == len(rollout.masks)):
            raise ValueError('actual prompt or rollout lengths changed')
        grammar = PrefixGrammar(self.language, self.mode)
        for token, symbol, mask in zip(rollout.tokens, rollout.symbols, rollout.masks):
            if mask != tuple(self.mapping[s] for s in grammar.allowed()) or self.mapping.get(symbol) != token:
                raise ValueError('actual grammar mask or token mapping changed')
            grammar.append(symbol)
        failure = None if grammar.complete else 'incomplete-prefix'
        try:
            expected_rule = grammar.finish() if grammar.complete else None
        except ValueError:
            expected_rule = None;failure = 'serialization-capacity'
        if (rollout.rule != expected_rule or rollout.failure != failure
                or (not grammar.complete and len(rollout.tokens) < self.language.max_tokens and grammar.allowed())):
            raise ValueError('unexpected stopping rule or parsed candidate')
        if not rollout.tokens:
            return (torch.zeros_like(fast.a), torch.zeros_like(fast.b)), dict(log_probability=0., replay_delta=0.)
        from .score_replay import prefix_score_gradient
        def selected(logits,j,token):
            permitted=rollout.masks[j]
            values=logits[list(permitted)].float().cpu()/self.temperature
            if not torch.isfinite(values).all():raise FloatingPointError('nonfinite proposal replay')
            return values.log_softmax(0)[permitted.index(token)]
        return prefix_score_gradient(fast,rollout.prompt,rollout.tokens,
            forward=lambda ids,pos:self._forward(ids,fast,pos),selected=selected,
            loss=arrived_loss,sampled_log_probability=rollout.sampled_log_probability,
            tolerance=self.replay_absolute_tolerance,on_backward=on_backward)
