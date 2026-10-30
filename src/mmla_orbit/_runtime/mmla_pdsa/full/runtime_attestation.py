"""Actual isolated-worker import and native-library provenance, no imports selected by data."""
from pathlib import Path
import hashlib,sys,sysconfig,platform,os,tempfile
from types import SimpleNamespace

_RETAINED=[]
_BOUND_NATIVE={}


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def retain_downward_library(manifest):
    """Keep original loader copies alive until the owning worker exits.

    Only fast_downward.interface's tempfile binding changes. Its original
    load_lib, ABI declarations and native bytes remain untouched.
    """
    import fast_downward.interface as interface
    source=Path(interface.DOWNWARD_LIB_PATH).resolve(strict=True)
    inventory={Path(r['path']).resolve():r['sha256'] for r in manifest['files']}
    if source not in inventory or _sha(source)!=inventory[source]:raise ValueError('pinned native loader source')
    class RetainedDirectory:
        def __enter__(self):
            self.directory=tempfile.TemporaryDirectory(prefix='bound-downward-')
            _RETAINED.append(self.directory)
            return self.directory.name
        def __exit__(self,kind,value,traceback):
            if kind is not None:return False
            path=(Path(self.directory.name)/'libdownward.so').resolve(strict=True)
            if path.is_symlink() or _sha(path)!=inventory[source]:raise ValueError('native copy differs from pinned source')
            st=path.stat()
            _BOUND_NATIVE[path]=dict(source=str(source),sha256=inventory[source],device=st.st_dev,inode=st.st_ino)
            return False
    interface.tempfile=SimpleNamespace(TemporaryDirectory=RetainedDirectory)


def attest(manifest):
    site=Path(manifest['site']).resolve();code=Path(manifest['code_root']).resolve()
    stdlib=Path(sysconfig.get_path('stdlib')).resolve();dyn=stdlib/'lib-dynload'
    expected=[str(site),str(code),str(stdlib),str(dyn)]
    if sys.path!=expected or not sys.flags.no_site or not sys.dont_write_bytecode:
        raise ValueError('private no-site worker import resolver differs')
    if platform.python_version()!='3.11.15' or platform.machine()!='x86_64':raise ValueError('fixed worker Python ABI')
    inventory={Path(r['path']).resolve():r['sha256'] for r in manifest['files']}
    source={code/rel:h for rel,h in manifest['source_files'].items()}
    rows=[]
    for name,module in sorted(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if not path or str(path).startswith('<'):continue
        p=Path(path).resolve(strict=True)
        if p in inventory:
            identity=_sha(p)
            if identity!=inventory[p]:raise ValueError('actual isolated imported package changed')
            origin='pinned_site'
        elif p in source:
            identity=_sha(p)
            if identity!=source[p]:raise ValueError('actual reviewed source changed')
            origin='reviewed_source'
        elif p.is_relative_to(stdlib) and not p.is_relative_to(stdlib/'site-packages'):
            identity=_sha(p);origin='actual_python_stdlib'
        else:raise ValueError('ambient import outside declared resolver: '+str(p))
        rows.append(dict(module=name,path=str(p),sha256=identity,origin=origin))
    native=[]
    mappings=[line.split(maxsplit=5) for line in Path('/proc/self/maps').read_text().splitlines()]
    for path in sorted({line.split(maxsplit=5)[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if len(line.split(maxsplit=5))==6 and line.split(maxsplit=5)[-1].startswith('/') and '.so' in line.split(maxsplit=5)[-1]}):
        p=Path(path).resolve(strict=True)
        binding=_BOUND_NATIVE.get(p)
        if binding is not None:
            st=p.stat();mapped=[r for r in mappings if len(r)==6 and r[5]==path]
            if ((st.st_dev,st.st_ino)!=(binding['device'],binding['inode']) or _sha(p)!=binding['sha256']
                    or not mapped or any(int(r[4])!=st.st_ino or tuple(int(x,16) for x in r[3].split(':'))!=(os.major(st.st_dev),os.minor(st.st_dev)) for r in mapped)):
                raise ValueError('actual native mapping differs from retained original copy')
        if not (binding is not None or p in inventory or p.is_relative_to(stdlib) or p.is_relative_to(Path('/usr/lib')) or p.is_relative_to(Path('/lib')) or p.is_relative_to(Path(sys.base_prefix)/'lib')):
            raise ValueError('native shared library outside fixed worker platform: '+str(p))
        native.append(dict(path=str(p),bytes=p.stat().st_size,sha256=_sha(p),retained_original=binding))
    required={'textworld','fast_downward','alfworld'}
    if not required.issubset(sys.modules):raise ValueError('required actual native modules absent')
    return dict(schema='orbit-worker-runtime-attestation-v1',python=platform.python_version(),SOABI=sysconfig.get_config_var('SOABI'),
        search_path=expected,imports=rows,native=native,
        scope='pinned package/source bytes and actual Python/platform library identities; retained per preflight, digested per episode')
