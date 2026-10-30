"""Hash-bound engineering views; fitting, calibration and final access stay distinct."""
from dataclasses import dataclass,asdict
import copy
import hashlib
import json
from pathlib import Path
from .profile import wire,digest
from .formatting import InputFormatter


def file_identity(path):
    p=Path(path);return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


@dataclass(frozen=True)
class EpisodeView:
    identity:str
    semantic_group:str
    split:str
    observations:dict
    support_prompt:str
    query_prompt:str
    support_authority:dict
    query_authority:dict
    limits:dict


class EngineeringDataset:
    """Annotations are held by the trusted loss/task adapter, not EpisodeView."""
    def __init__(self,source,compiler,*,source_manifest_sha256,compiler_manifest_sha256,tokenizer=None,tokenizer_identity=None,
                 family_manifest=None,family_manifest_sha256=None):
        self.source,self.compiler=Path(source),Path(compiler)
        for path,expected in ((self.source/'manifest.json',source_manifest_sha256),
                              (self.compiler/'manifest.json',compiler_manifest_sha256)):
            if type(expected) is not str or len(expected)!=64 or file_identity(path)['sha256']!=expected:
                raise ValueError('explicit immutable dataset manifests required')
        self.manifest=json.loads((self.source/'manifest.json').read_bytes())
        self.compiler_manifest=json.loads((self.compiler/'manifest.json').read_bytes())
        if (self.compiler_manifest['source_manifest_sha256']!=source_manifest_sha256
                or self.compiler_manifest['groups']!=self.manifest['groups']
                or self.manifest['generated_episodes']!=640):
            raise ValueError('derived/source group or source binding mismatch')
        self.family_manifest=None
        if (family_manifest is None)!=(family_manifest_sha256 is None):raise ValueError('complete family grouping binding')
        if family_manifest is not None:
            if file_identity(family_manifest)['sha256']!=family_manifest_sha256:raise ValueError('family manifest changed')
            fm=json.loads(Path(family_manifest).read_bytes())
            if (fm['schema']!='mmla-query-free-families-v1' or fm['source_manifest_sha256']!=source_manifest_sha256
                    or fm['compiler_manifest_sha256']!=compiler_manifest_sha256 or len(fm['mapping'])!=640):
                raise ValueError('derived family/source binding mismatch')
            order={'fit':0,'calibration':1,'final':2}
            for item in fm['mapping'].values():
                if (fm['families'][item['family']]!=item['split']
                        or self.manifest['groups'][item['semantic_group']]!=item['source_split']
                        or order[item['split']]<order[item['source_split']]):raise ValueError('holdout downgrade or inconsistent mapping')
            from .family_data import derive
            expected=derive(self.source,self.compiler,source_sha=source_manifest_sha256,compiler_sha=compiler_manifest_sha256)
            for field in ('schema','mapping','families','counts','rule','q_prefix_scope','source_presentations'):
                if fm[field]!=expected[field]:raise ValueError('family mapping does not follow the complete actual source inventory')
            self.family_manifest=fm
        self.identity=digest(dict(source=source_manifest_sha256,compiler=compiler_manifest_sha256,grouping=family_manifest_sha256))
        self._rows={};self._labels={};self._view_hash={};self.access_log=[]
        if (tokenizer is None)!=(tokenizer_identity is None):raise ValueError('complete formatter binding required')
        self.formatter=None if tokenizer is None else InputFormatter(tokenizer,tokenizer_identity=tokenizer_identity)

    def load(self,split,*,purpose,final_model_identity=None):
        allowed={'fit':'module-training','calibration':'fixed-calibration','final':'locked-final-evaluation'}
        if split not in allowed or purpose!=allowed[split]:raise PermissionError('split has a distinct declared purpose')
        if split=='final' and (type(final_model_identity) is not str or len(final_model_identity)!=64):
            raise PermissionError('final reader requires a locked model identity')
        rows=[];labels=[]
        source_splits=(split,) if self.family_manifest is None else ('fit','calibration','final')
        for source_split in source_splits:
            name=source_split+'.jsonl'
            for root,manifest,output in ((self.source,self.manifest,rows),(self.compiler,self.compiler_manifest,labels)):
                specs=[s for s in manifest['files'] if s['path']==name]
                if len(specs)!=1:raise ValueError('missing or duplicate split file')
                spec=specs[0];raw=(root/name).read_bytes()
                if len(raw)!=spec['bytes'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('dataset file identity mismatch')
                for line in raw.splitlines():
                    item=json.loads(line)
                    if self.family_manifest is not None:
                        key=item['id'] if root==self.source else item['source_id']
                        binding=self.family_manifest['mapping'][key]
                        checksum='source_row_sha256' if root==self.source else 'compiler_row_sha256'
                        if binding['source_split']!=source_split or digest(item)!=binding[checksum]:raise ValueError('mapped original row changed')
                        if binding['split']!=split:continue
                    output.append(item)
        if len(rows)!=len(labels):raise ValueError('missing compiler labels')
        seen=set();views=[]
        for row,label in zip(rows,labels):
            key=row['id'];group=row['semantic_group']
            if key in seen or (self.family_manifest is None and self.manifest['groups'].get(group)!=split):
                raise ValueError('duplicate source or split/group escape')
            seen.add(key)
            if (label['source_id']!=key or label['semantic_group']!=group
                    or label['source_row_sha256']!=hashlib.sha256(wire(row)).hexdigest()):
                raise ValueError('compiler target does not bind source row')
            self._rows[key]=copy.deepcopy(row);self._labels[key]=copy.deepcopy(label)
            task_group=group if self.family_manifest is None else self.family_manifest['mapping'][key]['family']
            view=EpisodeView(key,task_group,split,copy.deepcopy(row['observations']),
                row['support']['prompt'],row['query']['prompt'],copy.deepcopy(row['support']['authority']),
                copy.deepcopy(row['query']['authority']),copy.deepcopy(row['limits']))
            self._view_hash[key]=digest(asdict(view));views.append(view)
        self.access_log.append(dict(split=split,purpose=purpose,final_model_identity=final_model_identity,
                                   episodes=len(rows),groups=len({v.semantic_group for v in views})))
        return tuple(views)

    def _validate_view(self,view):
        if type(view) is not EpisodeView or self._view_hash.get(view.identity)!=digest(asdict(view)):
            raise PermissionError('observation/authority/view changed or source unopened')
        return self._rows[view.identity]

    def prompt_tokens(self,view,*,role):
        row=self._validate_view(view)
        if role not in ('support','query') or self.formatter is None:
            raise PermissionError('bound source formatter required')
        return self.formatter.prompt(row[role]['prompt'])

    def compiler_targets(self,view):
        self._validate_view(view)
        result=copy.deepcopy(self._labels[view.identity])
        result['source_semantic_group']=result['semantic_group']
        result['semantic_group']=view.semantic_group
        return result

    def release_feedback(self,view,*,problem):
        """Release the support answer after this task's actual pending attempt.

        The prompt is reconstructed from the stored support source through the
        bound formatter. Neural memory remains a separate input.
        """
        from .runtime import Problem
        row=self._validate_view(view)
        expected_prompt=self.prompt_tokens(view,role='support')
        if (type(problem) is not Problem or view.identity not in self._rows
                or problem.problem_id!=view.identity or problem.pending is None
                or problem.failed or problem.closed or problem.model.profile.memory_view!='neural'):
            raise PermissionError('feedback requires this task’s completed pending attempt')
        attempt=problem.pending
        if (type(expected_prompt) is not tuple or not expected_prompt or attempt.prompt!=expected_prompt
                or attempt.world_clock!=row['support']['authority']['clock']
                or problem.tenant!=row['support']['authority']['tenant']
                or problem.principal!=row['support']['authority']['principal']):
            raise PermissionError('support prompt or trusted read envelope mismatch')
        return copy.deepcopy(self._rows[view.identity]['support']['answer'])

    def query_target(self,view):
        self._validate_view(view)
        # Returned to the trusted loss/scorer only, never to the proposal prompt.
        return copy.deepcopy(self._rows[view.identity]['query']['answer'])
