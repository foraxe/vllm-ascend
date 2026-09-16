"""Temporary benchmark-only opt-in, with exact restoration of delivery source."""
import argparse,hashlib,os,tempfile
from pathlib import Path

BASE_SHA='a48ee4f7a26a1558e3c225b965f4b49d3b3c7b77459ee4134535795039dd9a18'
FOOTER=b'''
# Temporary native CPU-offload reference benchmark.
if _engram_vmm_os.environ.get('ENGRAM_NATIVE_CPU_REFERENCE') == '1':
    import sys as _engram_cpu_sys
    from reference import install as _install_engram_cpu_reference
    _install_engram_cpu_reference(_engram_cpu_sys.modules[__name__])
'''
p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--restore',action='store_true')
a=p.parse_args()
path=a.repo/'vllm_ascend/models/deepseek_v41/model.py'
data=path.read_bytes()
base=data[:-len(FOOTER)] if data.endswith(FOOTER) else data
if hashlib.sha256(base).hexdigest()!=BASE_SHA:
    raise RuntimeError('Unexpected model changes; refusing overwrite')
wanted=base if a.restore else base+FOOTER
if wanted!=data:
    fd,tmp=tempfile.mkstemp(prefix='.cpu-reference-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(wanted);f.flush();os.fsync(f.fileno())
        os.chmod(tmp,path.stat().st_mode&0o777)
        os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)
print('CPU_REFERENCE_RESTORED' if a.restore else 'CPU_REFERENCE_INSTALLED')
