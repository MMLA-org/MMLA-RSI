# Rules, state and admission

RTT contains three bounded prefix trees, producing a step size, clipping
threshold and momentum coefficient. The interpreter, projection and momentum
layout are fixed. RTMA contains one tree scoring NULL and complete-row slot
replacements. Candidate format, provenance and size are validated before slots
are enumerated. NULL wins ties. A joint program concatenates RTT then RTMA trees.

Every completed development attempt supplies a bounded loss/cost record. The
current parent, not the candidate under evaluation, updates the proposal learner.
RTT uses loss-times-score gradients over the actual grammar-masked sample;
incomplete candidate draws retain their probability. RTMA manages the records
available to the next proposal. Full archives are for selection/recovery, not an
extra proposal-history input. Each new attempt reconstructs the prompt and cache.

Only eligible attempts after the required parent adaptation enter selection.
Selection orders eligible attempts by highest reward, lowest cost, program
symbols, then attempt index; otherwise tied copies of the same program use the
earliest eligible attempt. A single locked
candidate is compared with the parent on its independent audit groups. Rejection
stops that finite run; acceptance changes the next generation's rule. An always
NULL manager is legal but produces no structurally eligible RTMA-only candidate.
Joint admission applies to the pair; it does not prove each component improved.

A score-gradient replay estimates the sampled candidate's development objective,
not an unbiased gradient through the entire selection/audit trajectory. Functional
participation and discovery efficiency require additional declared interventions.
The controller's `NOT_MEASURED` participation field is deliberate and must not be
turned into a causal-success flag by downstream reporting.
