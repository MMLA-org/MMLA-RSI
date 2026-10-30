"""Load the three closed R2 rho156 originals for the new proposal course.

The host authenticates the input-binding file and mounts only the three prior
portable pairs read-only. This boundary authenticates and decodes those exact
data-only artifacts. It never retrains, loads an optimizer, or resumes R2.
"""
from pathlib import Path

from mmla_pdsa.full.safe_state import exact_equal
from mmla_pdsa.full.extensions.rsi_binding import isolated_model, rebind, check_role_pair
from trained_pairs import identity, _share_equal
from reward_training import read_artifact
from curriculum import digest


def restore_initial_pairs(original_pairs, binding, *, directory, on_restored):
    if (tuple(original_pairs) != (17,29,43)
            or binding['schema'] != 'orbit-process-credit-existing-inputs-v1'
            or binding['status'] != 'EXISTING_BYTES_AND_TOKENIZER_VERIFIED'
            or [r['seed'] for r in binding['actual_initial_models']] != [17,29,43]):
        raise ValueError('one bound closed R2 input collection with all three seeds required')
    pairs = {}
    for row in binding['actual_initial_models']:
        seed = row['seed']; original = original_pairs[seed]
        if (original['seed'] != seed or original['objective'] != 'static'
                or int(original['task'].rho.version) != 942 or int(original['proposal'].rho.version) != 128):
            raise ValueError('authentic corrected task942/proposal128 dependency required')
        check_role_pair(original['task'], original['proposal'])
        before = {role:identity(original[role]) for role in ('task','proposal')}
        # Host paths from the metadata are never used as container read paths.
        payload = read_artifact(Path(directory)/('static-'+str(seed)), source=binding['old_source'],
            kind='task-return-trained-pair', manifest_sha256=row['manifest']['sha256'])
        if (payload['schema'] != 'orbit-return-trained-pair-v1'
                or payload['seed'] != seed or payload['source'] != binding['old_source']
                or payload['objective'] != 'static' or payload['original'] != before
                or payload['training_identity'] != row['training_identity']
                or digest(payload['training']) != row['training_identity']
                or payload['proposal_identity'] != row['identities']['proposal']
                or before['task'] != row['identities']['task']
                or payload['deployment_Phi'] != 'fixed' or payload['RSI_qualified'] is not False
                or row['identities']['proposal']['rho_version'] != 156
                or not exact_equal(payload['initialization'], original['proposal'].rho.initialization)):
            raise ValueError('closed R2 payload does not match bound model lineage')
        proposal = isolated_model(original['proposal'])
        proposal.restore_new_state(payload['proposal_modules'])
        proposal.module_status = dict(payload['module_status'])
        proposal.eval().requires_grad_(False); rebind(proposal)
        if (identity(proposal) != row['identities']['proposal']
                or proposal.module_status['rho'] != 'TRAINED_TASK_RETURN_EDIT_PROPOSAL'):
            raise ValueError('decoded rho156 differs from the actual R2 original')
        _share_equal(original['task'], proposal)
        if {role:identity(original[role]) for role in ('task','proposal')} != before:
            raise ValueError('loading R2 must preserve the earlier role originals')
        pairs[seed] = dict(task=original['task'],proposal=proposal,objective='static',seed=seed,
            input_source=binding['old_source'],input_training_identity=row['training_identity'])
        on_restored(dict(seed=seed,identities={r:identity(pairs[seed][r]) for r in ('task','proposal')},
            manifest_sha256=row['manifest']['sha256'],training_identity=row['training_identity'],
            original_R2_weights_restored=True,optimizer_restored=False,old_run_resumed=False))
    return pairs
