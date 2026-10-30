# Architecture runtime

Orbit separates the frozen language substrate, the trained initialization `rho`, a problem-private numerical state `Phi`, and authoritative memory `M`. Reset copies the trained initialization; restoring a saved session instead retains its adapted factors, momenta, memory and random state.

## Three source interfaces

| Interface | Operation | Existing implementation |
|---|---|---|
| Frame | Consume an authenticated complete command frame, construct and commit memory actions | `mmla_pdsa.full.frames.FrameDriver` |
| Stream | Consume the next source token, close a causal segment, retain bounded unfinished context, then process completed commands | `mmla_pdsa.full.extensions.stream.StreamDriver` |
| Joint | Seal arrived feedback and select a KEEP/UPDATE × memory-action pair at the next original event | `mmla_pdsa.full.extensions.joint.JointStreamDriver` |

The stream profile uses deterministic delimiter CSBC. The joint driver computes the proposed numerical update before selecting the pair, so choosing KEEP still incurs that computation. Its numerical and memory commits have separate ownership.

For typed engineering records, call `memory, reader, binding = orbit.new_engineering_memory()`, then pass that memory and reader to `orbit.new_problem(...)` and choose a driver. This restores the original independent frozen qualification decoder; it does not substitute the task initialization for that reader. The actual device and context profile create a fresh gate with an empty qualification cache. `binding` records the original and current identities. The frozen gate checks record reconstruction and its declared field queries. A rejected record cannot become a qualified neural memory row. Gate identity binds the actual profile, frozen decoder and encoding; qualification from an older model/profile is not silently inherited.

Source adapters own authenticated envelopes and the timing of external feedback. A model-generated answer is not its own ground-truth feedback. `EngineeringTask` and `EngineeringStreamTask` provide the existing versioned source adapter; callers must supply legally accessible original data. The package does not download task data or inject teacher rows into memory.

## Numerical feedback and recovery

`Problem.feedback` and `StreamDriver.feedback` update the private LARC factors from arrived supervision under an explicit bounded updater. `JointStreamDriver.arrive` seals the same feedback for the next joint event. Frozen substrate weights still carry the gradient to the input residual. Forward computation uses the saved eager/recompute arithmetic, with state-dependent caches rebuilt.

`mmla_pdsa.full.session.save_session/load_session` handles quiet Frame sessions. `mmla_pdsa.full.extensions.stream_session.save_stream_session/load_stream_session` handles quiet Stream and Joint sessions. Recovery preserves numeric state and authoritative row content while rebinding epochs and handles. The caller publishes the restored graph and applies the returned random state before further computation. In-flight operations are not replayed automatically.

## Predictive decisions and recursive management

The inherited Q/J heads are stored, but their historical calibration is not transferred to this changed reader or observation domain. Their qualification guards remain active. An explicit fixed selector can exercise legal issued actions; this is reported as nonpredictive and does not certify Q/J accuracy.

The released recursive controller follows the **RTMA route** described in `RSI_RUNTIME.md`: the current manager controls task observations and proposal evaluation memory; a successor replaces it only after independent admission. Numerical factors stay fixed in the completed recursive experiment. Numerical RTT and joint source interfaces are separate available execution paths; the experiment's RTMA evidence is not assigned to them.

The package verification record gives the actual mode execution and recovery status. The source code and small synthetic checks alone do not establish a measured task benefit.
