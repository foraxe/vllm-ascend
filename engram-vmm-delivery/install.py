"""Install only the opt-in model footer into the exact validated fork.

Keep this directory on PYTHONPATH. Stop serving workers before deployment.
Unknown model edits are refused, not overwritten. --restore returns the
original HBM source and leaves unrelated files untouched.
"""
import argparse,hashlib,os,subprocess,tempfile
from pathlib import Path

COMMIT='e43cf1e9f5d9bead076853aa6bcacb671465de94'
SOURCE_SHA='fe2e0603e672c13e931c24fb3545f821d904c5e5b618ccc6e52973cb4623edcf'
EXPERIMENT_SHA='096fb5a1ce2e32f9d1af9bc601a7ff39fe2951738139b379a6618f2b9a0953e1'
REL='vllm_ascend/models/deepseek_v41/model.py'
FOOTER='''
# Opt-in VMM Engram. No opt-in means the original HBM implementation.
import os as _engram_vmm_os
if _engram_vmm_os.environ.get('VLLM_ASCEND_ENGRAM_VMM_RUN'):
    import sys as _engram_vmm_sys
    from engram_vmm import install as _install_engram_vmm
    _install_engram_vmm(_engram_vmm_sys.modules[__name__])
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--check',action='store_true')
    p.add_argument('--restore',action='store_true')
    p.add_argument('--replace-experiment',action='store_true')
    a=p.parse_args()
    head=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()
    if head!=COMMIT: raise RuntimeError(f'Unsupported source revision: {head}')
    pristine=subprocess.check_output(['git','-C',str(a.repo),'show',f'HEAD:{REL}'])
    if hashlib.sha256(pristine).hexdigest()!=SOURCE_SHA:
        raise RuntimeError('Pristine model source checksum mismatch')
    target=a.repo/REL
    current=target.read_bytes()
    patched=pristine+FOOTER.encode()
    known_experiment=hashlib.sha256(current).hexdigest()==EXPERIMENT_SHA
    if current not in (pristine,patched) and not (known_experiment and a.replace_experiment):
        raise RuntimeError('Unknown/unapproved model edits; refusing overwrite')
    wanted=pristine if a.restore else patched
    if a.check:
        print('INSTALL_CHECK_PASS',target)
        return
    if known_experiment:
        backup=target.with_suffix('.py.engram-experiment.bak')
        if backup.exists():
            if backup.read_bytes()!=current: raise RuntimeError('Existing backup differs')
        else:
            with backup.open('xb') as f:f.write(current)
    if current!=wanted:
        fd,name=tempfile.mkstemp(prefix='.engram-model-',dir=target.parent)
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(wanted);f.flush();os.fsync(f.fileno())
            os.chmod(name,target.stat().st_mode & 0o777)
            os.replace(name,target)
        finally:
            Path(name).unlink(missing_ok=True)
    print('RESTORED_HBM' if a.restore else 'INSTALLED_OPT_IN',target)


if __name__=='__main__':main()
