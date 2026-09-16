"""Offline cleanup of one stopped run's capability files; never recursive.

Caller must prevent concurrent restarts/imports while this runs. Refuses if
any process in this PID namespace still maps the VMM library. Use only in the
dedicated serving container, not while another container can access this shm.
"""
import argparse,os,re
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--run',required=True)
a=p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',a.run):raise ValueError('Invalid run ID')
root=Path('/dev/shm')/('engram-vmm-'+a.run)
if root.is_symlink():raise RuntimeError('Refusing symlink run directory')
if not root.exists():print('ALREADY_CLEAN');raise SystemExit(0)
if root.stat().st_uid!=os.getuid() or root.stat().st_mode&0o077:
    raise RuntimeError('Run directory ownership/permissions mismatch')
for proc in Path('/proc').glob('[0-9]*'):
    try:
        maps=(proc/'maps').read_text()
        args=(proc/'cmdline').read_bytes().split(b'\0')
        comm=(proc/'comm').read_text()
    except (FileNotFoundError,ProcessLookupError):continue
    if 'libengram_host_vmm.so' in maps or comm.startswith('VLLM') or (b'serve' in args and b'/home/admin/model-csi/model' in args):
        # Zombies have no mapping and cannot access a handle.
        state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
        if state!='Z':raise RuntimeError(f'Worker/library still active: PID{proc.name}')
allowed={f'{i}.{plane}{suffix}' for i in (0,1) for plane in ('codes','scales') for suffix in ('','.ready')}
files=list(root.iterdir())
if any(x.name not in allowed or x.is_symlink() or not x.is_file() for x in files):
    raise RuntimeError('Unexpected run contents; refusing deletion')
for path in files:path.unlink()
root.rmdir()
print('CLEANED_STOPPED_RUN',a.run,'files',len(files))
