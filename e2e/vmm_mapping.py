"""Persistent shared host VMM allocation, adapted from Supermem M7."""
import ctypes
import os
from pathlib import Path
import torch
from test_shared_consumer import tensor_at

GIB=1<<30

class Allocation(ctypes.Structure):
    _fields_=[('host_ptr',ctypes.c_void_p),('device_ptr',ctypes.c_void_p),
              ('size',ctypes.c_size_t),('physical_device',ctypes.c_int32),
              ('owner',ctypes.c_int32),('host_observed_location_type',ctypes.c_int32),
              ('host_observed_location_id',ctypes.c_int32),
              ('device_observed_location_type',ctypes.c_int32),
              ('device_observed_location_id',ctypes.c_int32)]

class VmmMapping:
    def __init__(self,path,rows,width,dtype,group):
        self.rows,self.width,self.dtype=rows,width,dtype
        self.itemsize=torch.empty((),dtype=dtype,device='cpu').element_size()
        self.size=((rows*width*self.itemsize+GIB-1)//GIB)*GIB
        self.device=torch.device('npu',torch.npu.current_device())
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        if group.rank==0 and (path.exists() or Path(str(path)+'.ready').exists()):
            raise ValueError(f'Refusing to replace live/stale VMM handle: {path}')
        self.lib=ctypes.CDLL(str(Path(__file__).with_name('libengram_host_vmm.so')))
        self.lib.host_shared_registered_alloc.argtypes=[ctypes.c_size_t,ctypes.c_int32,ctypes.c_char_p,ctypes.c_int32,ctypes.POINTER(Allocation)]
        self.lib.host_shared_registered_free.argtypes=[ctypes.POINTER(Allocation)]
        self.lib.host_shared_last_error.restype=ctypes.c_char_p
        self.lib.host_vmm_lifecycle_counts.restype=ctypes.c_uint64
        self.allocation=Allocation()
        rc=self.lib.host_shared_registered_alloc(self.size,self.device.index,os.fsencode(path),int(group.rank==0),ctypes.byref(self.allocation))
        if rc: raise RuntimeError(f'Host VMM: {self.lib.host_shared_last_error().decode()}')
        self.tensor=tensor_at(self.allocation.device_ptr,(rows,width),dtype,self.device)
        chunk=1<<22
        self.ptrs=torch.tensor([self.tensor.data_ptr()+start*width*self.itemsize for start in range(0,rows,chunk)],dtype=torch.int64,device=self.device)
        print(f'HOST_VMM_MAPPED rank={group.rank} bytes={self.size} host_location={self.allocation.host_observed_location_type}:{self.allocation.host_observed_location_id} device_location={self.allocation.device_observed_location_type}:{self.allocation.device_observed_location_id}',flush=True)

    def publish(self,start,source):
        local=source.to(self.device)
        self.tensor[start:start+len(source)].copy_(local)

    def lifecycle_counts(self):
        value=self.lib.host_vmm_lifecycle_counts()
        return value>>32,value&0xffffffff

    def close(self):
        torch.npu.synchronize()
        self.tensor=None
        self.ptrs=None
        rc=self.lib.host_shared_registered_free(ctypes.byref(self.allocation))
        if rc: raise RuntimeError(f'VMM free rc={rc}')
