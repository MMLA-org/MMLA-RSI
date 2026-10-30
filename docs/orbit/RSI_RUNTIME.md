# Running RTMA and RSI with Orbit

The public controller uses one fixed Orbit model. It generates memory-manager edits, evaluates each edit on complete tasks, feeds those results back through bounded proposal memory, locks a candidate, and evaluates it on an independent audit. An accepted manager becomes the parent used by the next generation. A run can finish with no accepted manager.

`model.run_rsi(...)` executes this process; `new_proposal_process(...)` is the lower-level sampler. Neither entry trains the base or the saved residual initializations.

## Model and execution state

The release contains the selected task model and its two matched proposal roles. The three proposal controls are:

- `trained-keep`: the learned proposer, with feedback retained by the parent manager.
- `trained-frozen`: the same learned proposer, with new feedback excluded from proposal memory.
- `original-keep`: the original proposer, with the same parent-managed feedback interface.

Each process has four attempts. All three finish before any audit is requested. A candidate must meet the existing development-selection conditions. Only an accepted successor proceeds to the second generation. There are at most two generations; changing the model or increasing this schedule defines another experiment.

Admission uses 1,024 complete paired tasks, a material increment of 0.02, and the original e-value threshold of 720 at alpha = 0.025/18. **720 is an e-value threshold, not a required number of successful tasks.** Unused comparison slots are not reassigned. Each invocation binds its own run, runtime, environment and development population; this does not authorize unlimited statistical retries.

## Connect a task host

The package supplies the controller and real task evaluator. A deployment host supplies the authorized task source, environment processes and resource accounting. Task files and private generation/validation logs are not distributed with the model.

The host must provide:

| Argument | Required behavior |
|---|---|
| `dev` | A complete 32-world `ReadyBlock`, obtained through `support_readiness.load_ready` with its expected request, producer and seal. |
| `open_episode(permit, seed, on_work)` | Open a fresh, teacher-free `LogicalWorldEpisode` for the supplied permit. Use the provided `LogicalProcessBackend` and a bound `LogicalRuntime`; retain their source/runtime checks. |
| `prepare_audit(generation, cohort, seen)` | After the controller saves the complete candidate lock, obtain a fresh 1,024-world block from the same population. Bind it to the cohort identity, exclude already seen semantic tasks, and keep private plans and audit outcomes unavailable to the proposer. |
| `on_work`, `on_event` | For the default scalar evaluator, pass the concrete `proposal_role_common.WorkLog.work` bound method and its `event` method. A lambda, wrapper or other logger does not satisfy the task-accounting owner contract. The log lives outside the immutable model directory. |
| `resources_complete()` | Check that actual work has completed and the host's limits and accounting are satisfied. Return a Boolean; do not substitute a constant success value. |
| `task_backend` | Optional qualified parallel scalar-world evaluator. Its complete result is checked by the same evaluator. The default uses the exact scalar path. |

`logical_runtime.prepare_logical_runtime` binds an existing reviewed environment manifest to the packaged logical worker. `logical_backend.LogicalRuntime`, `logical_episode.LogicalWorldEpisode` and `logical_backend.LogicalProcessBackend` supply the process transport. They do not download or install dependencies. The environment additionally needs its declared TextWorld/ALFWorld/native planner dependencies; the inference-only `requirements.txt` is not an environment installation recipe.

The controller checks data and result identities. A Python callback is not an operating-system isolation boundary. For independent scientific claims, the host must enforce the declared data visibility, native resource envelope and independent support production, as well as preserve the final results.

Given those host bindings, create a `WorkLog` with the host's declared finite work limits. The host closes it during teardown, including failure cleanup.

```python
from pathlib import Path
from proposal_role_common import WorkLog

logs = Path("/path/outside/model/new-run-logs")
logs.mkdir(parents=True, exist_ok=False)
work = WorkLog(logs, phase="RTMA-deployment", limits=declared_work_limits)

result = model.run_rsi(
    directory="/path/outside/model/new-run",  # must not already exist
    run_identity=run_contract_sha256,
    environment_identity=environment_contract_sha256,
    dev=dev,
    open_episode=open_episode,
    prepare_audit=prepare_audit,
    on_work=work.work,
    on_event=work.event,
    resources_complete=accounting_boundary,
    task_backend=qualified_parallel_backend,
)
```

The new invocation has a separate test-family identity. The original model-training and historical result identities remain unchanged. This entry never calls the workflow that trains the proposal role first.

## Recovery and installed rules

The controller saves all proposal states, complete cohort records and the installed rule journal. Its closed artifact binds the complete result and the exact journal cut:

```python
saved = result["controller_artifact"]
recovered = model.restore_rsi(
    "/path/outside/model/new-run",
    manifest_sha256=saved["manifest_sha256"],
)
manager = recovered.manager
state = recovered.new_task_state(seed=123, tenant="example")
```

If an independent successor was accepted, `manager` is that installed rule. Otherwise it is the original parent. The task's numerical state starts at the saved trained initialization. Recovery does not rerun tasks or turn a rejected candidate into an accepted one.

Closed-controller recovery requires complete resource accounting. Interrupted runs retain their partial artifacts and cannot be restarted by reusing their output directory. The separate `save_proposal` / `restore_proposal` API supports quiet feedback-to-next-proposal recovery, including retained M and the next attempt's random tape. These two recovery scopes are distinct.

## Other architecture modes

`model.new_task_state(seed=..., tenant=..., policy="fixed_rho")` provides the observation-memory task state used by this RTMA route. `policy="adaptive"` enables the existing numerical-feedback path with the bound updater and manager. An optional typed rule binds a replacement updater, manager or pair. This is an execution capability; the fixed-Phi experiment does not provide a performance result for that adaptive mode.

`model.new_problem(memory, problem_id=..., tenant=..., principal=..., seed=...)` binds the loaded task model to an authoritative `Memory` with matching profile. The packaged `FrameDriver`, `StreamDriver` and `JointStreamDriver` provide complete-command, deterministic delimiter-CSBC stream, and joint state interfaces. Stream use requires explicit boundary/limits and selector identities. Learned frontend, construction and neural read modules remain the saved modules. A predictive J selector still requires qualification for its actual reader, memory domain and task; an unrelated old qualification is not inherited.

The observed RSI experiment uses RTMA with fixed numerical state. It does not run all optional source transports in that one trajectory, nor establish their joint benefit. The release validation report distinguishes available interfaces, actual executions and scientific results.
