# Orbit: loading, feedback and recovery

Orbit has three separate numerical roles within one selected model: the task reader at rho942, the process-credit-trained proposal role at rho184, and the original proposal role at rho156. They share equal frozen computation and retain private residual factors and initializations. The task reader stays fixed in this RTMA experiment.

The model jointly chooses an edit location and a bounded replacement subtree, producing edits to the current memory-management program. Completed evaluation feedback becomes an authoritative proposal record; the current manager decides which records remain readable by the next proposal. A process has four attempts, with at most three intervening memory commits. The retained records are rebuilt into each input; an old prompt or cache does not keep evicted records readable.

## Loading

Use a fresh Python process with the matching installed `mmla-rsi[orbit]` runtime. The package includes its original frozen base, always-on omega and the three role states of selected training seed17. The default loader selects that model.

```python
import torch
from mmla_orbit import load_orbit

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
model = load_orbit("/path/to/model", device="cuda")
```

Loading checks the package and role identities, preserves the global torch RNG, and performs no inference, training, download or rule admission. The task model is available as `model.task`; its explicit MMLA state interfaces remain those of the supplied runtime. This loader uses the declared PyTorch BF16/eager arithmetic and requires its recorded library versions.

For complete task evaluation, independent admission, manager succession and controller recovery, see [RSI_RUNTIME.md](RSI_RUNTIME.md).

## Proposal, feedback and recovery

```python
process = model.new_proposal_process(process="trained-keep", generation=0)
draw = process.draw()
candidate_symbols = draw.sample.program.symbols
# Evaluate that candidate with the declared task evaluator, then supply its
# actual complete-task mean loss and normalized cost:
process.feedback(draw, loss=observed_loss, cost=observed_cost, tag="completed")

saved = model.save_proposal(process, "/path/outside/model/proposal-checkpoint")
restored = model.restore_proposal(
    saved["path"], manifest_sha256=saved["manifest_sha256"],
    process="trained-keep", generation=0,
)
next_draw = restored.draw()
```

Save between feedback and the next proposal. Recovery reconstructs the committed memory transitions and the next attempt's independent random tape; it makes no forward call. Pending or failed attempts remain archived. Checkpoints belong outside the immutable model package.

The matched process choices are `trained-keep`, `trained-frozen`, and `original-keep`. Frozen means that new proposal feedback is not retained in M; it still uses the trained proposal role. The original comparison uses rho156 and the same external random-tape rule. The current experiment supplies generation indices 0 and 1. To instantiate an admitted successor in generation 1, provide its exact `parent_symbols` both when creating and restoring that process.

A supplied program is not certified by this loading API. Independent task evaluation, the fixed acceptance rule, and actual parent-to-successor transition records establish an accepted edge. The containing result report states which edges, if any, were accepted. Loading or changing M does not itself establish RSI benefit.

## Package identity

`MANIFEST.json` binds model-package files by relative path, size and SHA256. The installed runtime has its own `runtime-manifest.json`; its identity must match `orbit_config.json`. No Python source is loaded from the weights directory. `orbit_config.json` uses `mmla-orbit-process-credit-config-v1`; it binds the base/omega recipe, original profile and hybrid configuration, source and collection, exact proposal interface and initial manager, and its one `roles["17"]` entry. The entry pins its `ROLES.json` byte hash and original training-receipt identity. `roles/<seed>/` uses `orbit-process-credit-public-roles-v2` and retains the completed new course identity: 28 effective updates, 29 physical updates including the save/restore comparison, and 116 candidate gradient accumulations per seed. Actual numerical and resource counts come from the saved run.

The model card identifies the selected weights and separates their results from earlier internal checkpoints.

## References and license

- MMLA: https://arxiv.org/abs/2606.28876v4
- LARC: https://arxiv.org/abs/2609.40063
- MMLA-RSI: https://arxiv.org/abs/2610.20294v1

MMLA runtime code is AGPL-3.0. Preserve the original MiniCPM acknowledgment, base-model license and notices in the containing package.

Role metadata version2 preserves the original training receipt identity and reports retained,segment,and cumulative work separately. Public retained bindings contain only source/content hashes;loading needs no private training directory. This metadata records continuation,not a new scientific qualification.

## Shared readout calibration

A model package may include `calibration/readout.json` and its safetensors
payload. Its configuration names the completed training course, source and
readout version. Loading first verifies the original three-role bundle and then
applies the four shared input-memory readout tensors. The saved task, proposal
and original-proposal residual initializations stay unchanged.

The update is a model parameter artifact, not a runtime option to enable extra
training. It has its own resulting role identities and does not transfer an old
RSI acceptance or predictive-selector qualification. Packages without the field
retain their original load path. Install the exact runtime revision declared by
the package; old packages do not silently adopt a new implementation.
