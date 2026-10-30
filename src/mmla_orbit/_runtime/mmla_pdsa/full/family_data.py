"""Conservative grouping of existing sources; no generation or model access."""
import argparse,copy,hashlib,json
from pathlib import Path
from .profile import wire,digest

SPLITS=('fit','calibration','final')


def family_spec(spec):
    base=copy.deepcopy(spec)
    base.pop('query');base.pop('support_query')
    def rename(word):return '_'.join('b' if w=='a' else 'a' if w=='b' else w for w in word.split('_'))
    swapped=copy.deepcopy(base)
    swapped['operations']=[rename(x) for x in base['operations']]
    swapped['initial']={rename(k):v for k,v in base['initial'].items()}
    return min((base,swapped),key=wire)


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(root,manifest):
    result={}
    for split in SPLITS:
        spec=next(x for x in manifest['files'] if x['path']==split+'.jsonl')
        raw=(root/spec['path']).read_bytes()
        if len(raw)!=spec['bytes'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('changed source file')
        result[split]=[json.loads(line) for line in raw.splitlines()]
    return result


def derive(source,compiler,*,source_sha,compiler_sha):
    source,compiler=Path(source),Path(compiler)
    if file_sha(source/'manifest.json')!=source_sha or file_sha(compiler/'manifest.json')!=compiler_sha:raise ValueError('manifest mismatch')
    sm=json.loads((source/'manifest.json').read_bytes());cm=json.loads((compiler/'manifest.json').read_bytes())
    if cm['source_manifest_sha256']!=source_sha:raise ValueError('compiler lineage mismatch')
    rows=read_source(source,sm);labels=read_source(compiler,cm);families={};mapping={}
    for split in SPLITS:
        if len(rows[split])!=len(labels[split]):raise ValueError('missing derived rows')
        for row,label in zip(rows[split],labels[split]):
            if (row['id'] in mapping or label['source_id']!=row['id']
                    or label['source_row_sha256']!=digest(row)
                    or sm['groups'][row['semantic_group']]!=split
                    or label['semantic_group']!=row['semantic_group']):raise ValueError('source binding mismatch')
            f=digest(family_spec(row['semantic_spec']))
            families[f]=max(families.get(f,'fit'),split,key=SPLITS.index)
            mapping[row['id']]=dict(family=f,source_split=split,semantic_group=row['semantic_group'],
                source_row_sha256=digest(row),compiler_row_sha256=digest(label))
    for item in mapping.values():item['split']=families[item['family']]
    if len(mapping)!=640 or len(families)!=80:raise ValueError('fixed source inventory changed')
    for f in families:
        if sum(x['family']==f for x in mapping.values())!=8:raise ValueError('eight actual presentations per family required')
    counts={s:dict(families=sum(v==s for v in families.values()),presentations=sum(v['split']==s for v in mapping.values())) for s in SPLITS}
    if counts!=dict(fit=dict(families=31,presentations=248),calibration=dict(families=17,presentations=136),final=dict(families=32,presentations=256)):
        raise ValueError('reviewed finite grouping changed')
    return dict(schema='mmla-query-free-families-v1',source_manifest_sha256=source_sha,
        compiler_manifest_sha256=compiler_sha,source_code_sha256=file_sha(__file__),
        rule='remove query/support_query, renormalize a/b; inherit final > calibration > fit',
        q_prefix_scope='non-initialization closed segments; four futures from one physical snapshot',
        mapping=mapping,families=families,counts=counts,source_presentations=640,
        independence='80 structural families, not 640 independent tasks',model_calls=0,new_source_tasks=0)


def build(source,compiler,destination,*,source_sha,compiler_sha):
    p=Path(destination)
    if p.exists():raise FileExistsError('immutable derived grouping')
    result=derive(source,compiler,source_sha=source_sha,compiler_sha=compiler_sha)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as file:file.write(wire(result)+b'\n')
    return dict(path=str(p),sha256=file_sha(p),bytes=p.stat().st_size,counts=result['counts'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--compiler',required=True)
    p.add_argument('--destination',required=True);p.add_argument('--source-sha',required=True);p.add_argument('--compiler-sha',required=True)
    a=p.parse_args();print(json.dumps(build(a.source,a.compiler,a.destination,source_sha=a.source_sha,compiler_sha=a.compiler_sha)))
