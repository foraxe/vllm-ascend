"""Validate original CPU lookup + pinned staging/H2D without tuning it."""
from types import SimpleNamespace
import torch,torch_npu
from reference import CPUOffloadEngram

torch.npu.set_device(0)
q=SimpleNamespace(size=1,rank=0)
t=CPUOffloadEngram(4097,256,q,device='npu',storage_format='int8')
codes=(torch.arange(4097*256,dtype=torch.int32).reshape(4097,256)%255-127).to(torch.int8)
scales=(torch.arange(4097*8).reshape(4097,8)%7+1).float()/17
t.weight.data.copy_(codes);t.weight_scale.copy_(scales)
for count in (0,1,7,24,3072,24,24,3,0):
    ids=(torch.arange(count,dtype=torch.int64)*131)%4097
    result=t.lookup_local(ids,pin_output=True)
    assert result.device.type=='cpu'
    if count:assert result.is_pinned()
    out=result.to('npu',non_blocking=True)
    t._record_offload_use(result.data_ptr(),torch.device('npu',0))
    ref=(codes[ids].float().view(count,8,32)*scales[ids,:,None]).reshape(count,256).bfloat16()
    assert torch.equal(out.cpu(),ref),count
torch.npu.synchronize()
print('NATIVE_CPU_REFERENCE_PASS',flush=True)
