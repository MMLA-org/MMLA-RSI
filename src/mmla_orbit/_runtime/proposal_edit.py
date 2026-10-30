"""Learned edit-site/subtree actions and prefix-exact slow-rho replay.

The native owner supplies one verified language/token codec/model version and
current-memory prompt builder. This module does not load models, run tasks,
update weights, perform admission, or read candidate archives. It is shared by
learned and frozen proposal controls; no useful-edit filter is present.
"""
from dataclasses import dataclass, replace
import math
from typing import Callable, Mapping

import torch
from content_rules import Language, PrefixGrammar, Rule, OPS, parse_rule


@dataclass(frozen=True)
class Site:
    start: int
    stop: int
    depth: int
    language: Language


class EditPlan:
    """Validate the external parent once; retain typed legal edit locations."""
    def __init__(self, language: Language, parent: Rule):
        if parent.mode != 'rtma' or parent.language != language.identity:
            raise ValueError('one bound RTMA parent required')
        if parse_rule(language, 'rtma', parent.symbols) != parent:
            raise ValueError('canonical parent program required')
        self.language, self.parent = language, parent
        depths, pending = [], [0]
        for symbol in parent.symbols:
            depth = pending.pop()
            depths.append(depth)
            if symbol in OPS:
                pending.extend((depth + 1, depth + 1))
        stops, stack = {}, []
        for start in range(len(parent.symbols) - 1, -1, -1):
            if parent.symbols[start] in OPS:
                stack.pop()  # left subtree
                stop = stack.pop()  # right subtree end
            else:
                stop = start + 1
            stops[start] = stop
            stack.append(stop)
        sites = []
        complete_cap = min(language.max_nodes, language.max_tokens, language.max_instructions)
        for start, depth in enumerate(depths):
            budget = complete_cap - (len(parent.symbols) - (stops[start] - start))
            subtree = replace(language, max_nodes=budget, max_tokens=budget,
                              max_instructions=budget, max_depth=language.max_depth - depth)
            sites.append(Site(start, stops[start], depth, subtree))
        self.sites = tuple(sites)

    def assemble(self, site: Site, grammar: PrefixGrammar) -> Rule:
        # The generated wire crosses into the existing interpreter here.
        if not grammar.complete:
            raise ValueError('an incomplete edit is not an executable program')
        symbols = self.parent.symbols[:site.start] + tuple(grammar.symbols) + self.parent.symbols[site.stop:]
        return parse_rule(self.language, 'rtma', symbols)


@dataclass(frozen=True)
class SampledEdit:
    parent: str
    site_index: int
    program: Rule
    symbols: tuple[str, ...]
    prompt: tuple[int, ...]
    tokens: tuple[int, ...]
    masks: tuple[tuple[int, ...], ...]
    site_prompt: tuple[int, ...]
    site_mask: tuple[int, ...]
    site_token: int
    token_log_probability: float
    site_log_probability: float
    rng_before: torch.Tensor
    rng_after: torch.Tensor


def sample_edit(model, plan: EditPlan, *, mapping: Mapping[str, int],
                site_prompt: tuple[int, ...], site_tokens: tuple[int, ...],
                prompt_for_site: Callable[[int], tuple[int, ...]],
                generator: torch.Generator, on_work: Callable, on_progress: Callable | None = None) -> SampledEdit:
    """Sample the full declared distribution. The caller owns rho until replay.

    A fresh, explicitly seeded CPU generator belongs to this attempt. Equal
    attempt seeds pair the controls despite earlier variable rollout lengths.
    The native owner saves an incomplete call and its advanced RNG on failure.
    """
    before = generator.get_state().clone()
    site_mask = site_tokens[:len(plan.sites)]
    if not site_prompt or len(site_prompt) + 1 > model.profile.local_tokens:
        raise ValueError('whole current-memory site prompt must fit; no truncation')
    device = model.rho.a.device
    if on_progress is not None:
        on_progress(dict(stage='site', site_prompt=site_prompt, site_mask=site_mask,
                         rng=before.clone()))
    with torch.no_grad():
        ids = torch.tensor([site_prompt], device=device, dtype=torch.long)
        position = torch.tensor([len(site_prompt)-1], device=device, dtype=torch.long)
        on_work(dict(kind='site_sample_forward', input_tokens=len(site_prompt), positions=1))
        scores = model.forward_batch(ids, torch.ones_like(ids), fast_states=None,
            memory=None, logit_positions=position)[0, 0, list(site_mask)].float().cpu()
        if not torch.isfinite(scores).all():
            raise FloatingPointError('nonfinite site logits')
        site_logp = scores.log_softmax(0)
        index = int(torch.multinomial(site_logp.exp(), 1, generator=generator))
        site_log_probability = float(site_logp[index])
    site = plan.sites[index]
    prompt = prompt_for_site(index)
    if not prompt or len(prompt) + site.language.max_tokens > model.profile.local_tokens:
        raise ValueError('whole current-memory prompt and edit must fit; no truncation')
    grammar = PrefixGrammar(site.language, 'rtma')
    tokens, masks, log_probability = [], [], 0.0
    if on_progress is not None:
        on_progress(dict(stage='subtree', site_index=index, site_prompt=site_prompt,
                         site_mask=site_mask, site_token=site_mask[index],
                         site_log_probability=site_log_probability, prompt=prompt, tokens=(), masks=(),
                         rng=generator.get_state().clone()))
    with torch.no_grad():
        while not grammar.complete:
            allowed = grammar.allowed()
            if not allowed:
                raise RuntimeError('declared edit grammar cannot complete within its budget')
            mask = tuple(mapping[symbol] for symbol in allowed)
            sequence = prompt + tuple(tokens)
            ids = torch.tensor([sequence], device=device, dtype=torch.long)
            position = torch.tensor([len(sequence) - 1], device=device, dtype=torch.long)
            on_work(dict(kind='edit_sample_forward', input_tokens=len(sequence), positions=1))
            logits = model.forward_batch(ids, torch.ones_like(ids), fast_states=None,
                                         memory=None, logit_positions=position)[0, 0]
            scores = logits[list(mask)].float().cpu()
            if not torch.isfinite(scores).all():
                raise FloatingPointError('nonfinite edit logits')
            logp = scores.log_softmax(0)
            selected = int(torch.multinomial(logp.exp(), 1, generator=generator))
            tokens.append(mask[selected]); masks.append(mask)
            log_probability += float(logp[selected])
            grammar.append(allowed[selected])
            if on_progress is not None:
                on_progress(dict(stage='subtree', site_index=index, site_prompt=site_prompt,
                    site_mask=site_mask, site_token=site_mask[index], site_log_probability=site_log_probability,
                    prompt=prompt, tokens=tuple(tokens),
                    masks=tuple(masks), symbols=tuple(grammar.symbols),
                    token_log_probability=log_probability, rng=generator.get_state().clone()))
    return SampledEdit(plan.parent.identity, index, plan.assemble(site, grammar),
        tuple(grammar.symbols), prompt, tuple(tokens), tuple(masks),
        site_prompt, site_mask, site_mask[index], log_probability,
        site_log_probability, before, generator.get_state().clone())


def action_logprobs(model, sample: SampledEdit, *, on_work: Callable):
    """Yield exact prefix log-probabilities for training and restore comparison."""
    device = model.rho.a.device
    prefixes = [(sample.site_prompt, sample.site_token, sample.site_mask, 'site')]
    prefixes += [(sample.prompt + sample.tokens[:j], token, mask, 'edit')
                 for j, (token, mask) in enumerate(zip(sample.tokens, sample.masks))]
    for sequence, token, mask, stage in prefixes:
        ids = torch.tensor([sequence], device=device, dtype=torch.long)
        position = torch.tensor([len(sequence)-1], device=device, dtype=torch.long)
        on_work(dict(kind=stage+'_replay_forward', input_tokens=len(sequence), positions=1))
        logits = model.forward_batch(ids, torch.ones_like(ids), fast_states=None,
            memory=None, logit_positions=position)[0, 0]
        scores = logits[list(mask)].float().cpu()
        if not torch.isfinite(scores).all():
            raise FloatingPointError('nonfinite replay logits')
        yield stage, scores.log_softmax(0)[mask.index(token)].double()


def replay_objective(model, sample: SampledEdit, advantage: float, *, on_work: Callable, backward_scale=None):
    """Replay the exact sampled site and subtree prefixes into slow rho.

    The cohort authenticates sample ownership and unchanged rho. Each prefix
    uses the sampler's selected-position forward and CPU FP32 normalization.
    The optimizer may step only after all sampled probabilities match.
    """
    if not math.isfinite(advantage) or not -1.0 <= advantage <= 1.0:
        raise ValueError('complete-task return-to-go must be finite in [-1,1]')
    if backward_scale is not None and (not math.isfinite(backward_scale) or backward_scale <= 0):
        raise ValueError('finite positive replay accumulation scale required')
    values = []
    for stage, value in action_logprobs(model, sample, on_work=on_work):
        if backward_scale is None:
            values.append(value)
        else:
            # Commit no optimizer state here. Release each prefix graph after
            # its contribution so a 31-token edit does not retain 31 trunks.
            (-advantage * backward_scale * value).backward()
            on_work(dict(kind=stage+'_replay_backward', positions=1))
            values.append(value.detach())
            del value
    site_logp = values[0]
    token_logp = torch.stack(values[1:]).sum()
    site_delta = abs(float(site_logp.detach()) - sample.site_log_probability)
    token_delta = abs(float(token_logp.detach()) - sample.token_log_probability)
    if max(site_delta, token_delta) > 1e-4:
        raise ValueError('sampled and replayed joint action probability differ')
    return -advantage * (site_logp + token_logp), dict(
        token_log_probability=float(token_logp.detach()), site_log_probability=float(site_logp.detach()),
        joint_log_probability=float((site_logp + token_logp).detach()),
        replay_delta=token_delta, site_replay_delta=site_delta)


def program_first_distribution(root_distribution: torch.Tensor, *, root_probability, root_index: int):
    """Marginalize learned edit positions, including aliases of full programs.

    Input is the softmax over the validated root-edit mask, not a conditional
    distribution at the sampled non-root location. Non-root edits keep the
    original first symbol. The caller binds identical masks in both memories.
    """
    result = root_distribution * root_probability
    result = result.clone()
    result[root_index] += 1.0 - root_probability
    return result


def batch_task_indices(epoch: int) -> tuple[tuple[int, ...], ...]:
    """Fourteen mixed-stratum batches; flatten input as stratum*4 + member."""
    if epoch not in (0, 1):
        raise ValueError('exactly two predeclared epochs')
    strata = range(14) if epoch == 0 else range(13, -1, -1)
    flat = tuple(4 * group + (member + epoch) % 4 for member in range(4) for group in strata)
    return tuple(flat[start:start + 4] for start in range(0, 56, 4))
