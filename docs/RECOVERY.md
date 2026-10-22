# Quiet recovery and serving versions

`Journal` writes bounded JSON/safetensors snapshots at complete quiet boundaries.
It checks manifests and contiguous journal state; tampering or a stale controller
fails closed. It does not checkpoint an arbitrary in-flight forward, task process
or partially committed update. Preserve failed attempts and their resource costs.
Resume only the latest complete validated boundary with the same configuration.

`ParentRegistry` records admitted program versions and pins a request to the
version it acquired. New requests can use a newly published version while
existing requests retain their original one. A registry publication does not
install weights or verify an external serving process; the application owns that
integration. Frozen reader bindings are local identity checks, not certificates
that can be transferred to a different process.

Storage limits and cooperative meters are library checks. They cannot stop a
hung third-party callback; use a supervisor for hard wall time, memory and process
limits. The library itself needs no administrator rights or system service.
