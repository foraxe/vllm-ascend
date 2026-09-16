"""Chunked shared-host mappings for full-size DSV4.1 Engram tables."""
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from safetensors import safe_open
from vllm.triton_utils import tl, triton
from vllm_ascend.models.deepseek_v41.engram_hbm import NodeShardedEngram, quantize_engram_rows
from test_shared_consumer import tensor_at

CHUNK_ROWS = 1 << 22  # 1GiB INT8 codes or 128MiB FP32 scales.


class Mapping:
    def __init__(self, path, rows, width, dtype, group):
        self.rows, self.width, self.dtype = rows, width, dtype
        self.itemsize = torch.empty((), dtype=dtype, device="cpu").element_size()
        size = ((rows*width*self.itemsize + 65535)//65536)*65536
        if group.rank == 0:
            if os.environ.get("ENGRAM_E2E_REUSE") == "1":
                if os.stat(path).st_size != size:
                    raise ValueError(f"Shared table reuse size mismatch: {path}")
            else:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                os.ftruncate(fd, size)
                os.close(fd)
        dist.barrier(group=group.cpu_group)
        self.fd = os.open(path, os.O_RDWR)
        self.mm = mmap.mmap(self.fd, size)
        self.host = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        if self.host%65536:
            raise ValueError('Host mapping must start at a 64KiB boundary')
        self.lib = ctypes.CDLL("libascendcl.so")
        self.lib.aclrtHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self.aliases, self.host_chunks = [], []
        device = torch.device("npu", torch.npu.current_device())
        if os.environ.get("ENGRAM_E2E_BOUNDED") == "1":
            self.ptrs = torch.empty(0,dtype=torch.int64,device=device)
            return
        for start in range(0, rows, CHUNK_ROWS):
            n = min(CHUNK_ROWS, rows-start)
            offset = start*width*self.itemsize
            count = min(CHUNK_ROWS*width*self.itemsize, size-offset)
            address = ctypes.c_void_p()
            rc = self.lib.aclrtHostRegister(self.host+offset, count, 0, ctypes.byref(address))
            if rc:
                raise RuntimeError(f"HostRegister {path} chunk={start} bytes={count} rc={rc}")
            self.host_chunks.append(self.host+offset)
            self.aliases.append(tensor_at(address.value, (n,width), dtype, device))
        self.ptrs = torch.tensor([x.data_ptr() for x in self.aliases], dtype=torch.int64, device=device)

    def cpu_view(self, start, end):
        return torch.frombuffer(self.mm, dtype=self.dtype, count=(end-start)*self.width,
                                offset=start*self.width*self.itemsize).view(end-start,self.width)

    def publish(self, start, source):
        if not self.aliases:
            self.cpu_view(start,start+source.shape[0]).copy_(source.cpu())
            return
        source = source.to(self.aliases[0].device)
        done = 0
        while done < source.shape[0]:
            chunk, local = divmod(start+done, CHUNK_ROWS)
            n = min(source.shape[0]-done, self.aliases[chunk].shape[0]-local)
            self.aliases[chunk][local:local+n].copy_(source[done:done+n])
            done += n


class SharedEngram(NodeShardedEngram):
    counter = 0

    def __init__(self, rows, width, query_group, device=None, storage_format="bf16", **kwargs):
        if storage_format != "int8" or width != 256:
            raise ValueError("E2E direct path requires INT8 Engram width256")
        self.control_resident = os.environ.get("ENGRAM_E2E_MODE") == "switch"
        init_device = torch.device("npu",torch.npu.current_device()) if self.control_resident else "meta"
        super().__init__(rows,width,query_group,device=init_device,storage_format=storage_format)
        index = SharedEngram.counter
        SharedEngram.counter += 1
        base = Path("/dev/shm") / f"engram-{os.environ['ENGRAM_E2E_RUN']}-{index}"
        self.ready_path=Path(str(base)+'.ready.json')
        root=Path('/home/admin/model-csi/model').resolve()
        self.identity={'root':str(root),'rows':rows,'width':width,
                       'index_sha256':hashlib.sha256((root/'quant_model_weights.safetensors.index.json').read_bytes()).hexdigest(),
                       'config_sha256':hashlib.sha256((root/'config.json').read_bytes()).hexdigest()}
        if os.environ.get('ENGRAM_E2E_REUSE')=='1':
            if not self.ready_path.exists() or json.loads(self.ready_path.read_text())!=self.identity:
                raise ValueError('Shared table lacks a matching completed-load record')
        self.codes_map = Mapping(str(base)+".codes",rows,width,torch.int8,query_group)
        self.scales_map = Mapping(str(base)+".scales",rows,width//32,torch.float32,query_group)
        # CPU parameter views retain correct checkpoint shapes without an HBM
        # table allocation. Inference reads the registered device aliases.
        if not self.control_resident:
            self.weight = torch.nn.Parameter(self.codes_map.cpu_view(self.start,self.end),requires_grad=False)
            self.weight_scale = self.scales_map.cpu_view(self.start,self.end)
        self.loaded = False
        print(f"ENGRAM_SHARED_MAP index={index} rank={query_group.rank} rows={rows} chunks={len(self.codes_map.aliases)}",flush=True)

    def load_checkpoint(self, model_path, key, chunk_rows=65536):
        if os.environ.get("ENGRAM_E2E_REUSE") == "1" and not self.control_resident:
            self.loaded=True
            print(f"ENGRAM_SHARED_REUSED rank={self.query_group.rank} key={key}",flush=True)
            return
        if self.control_resident:
            NodeShardedEngram.load_checkpoint(self,model_path,key,chunk_rows)
            if os.environ.get('ENGRAM_E2E_REUSE')=='1':
                self.loaded=True
                print(f'ENGRAM_SHARED_REUSED rank={self.query_group.rank} key={key} control_resident=True',flush=True)
                return
            for start in range(self.start,self.end,chunk_rows):
                stop=min(start+chunk_rows,self.end)
                self.codes_map.publish(start,self.weight.data[start-self.start:stop-self.start])
                self.scales_map.publish(start,self.weight_scale[start-self.start:stop-self.start])
            torch.npu.synchronize()
            self.loaded=True
            print(f"ENGRAM_SHARED_LOADED rank={self.query_group.rank} key={key} control_resident=True",flush=True)
            return
        root = Path(model_path)
        index = json.loads((root/"quant_model_weights.safetensors.index.json").read_text())["weight_map"]
        with safe_open(root/index[key],framework="pt",device="cpu") as f:
            data = f.get_slice(key)
            if data.get_shape() != [self.rows,self.width]:
                raise ValueError(f"Engram source shape mismatch: {key}")
            if data.get_dtype() == "BF16":
                for start in range(self.start,self.end,chunk_rows):
                    stop = min(start+chunk_rows,self.end)
                    codes,scales = quantize_engram_rows(data[start:stop].to(self.codes_map.ptrs.device))
                    self.codes_map.publish(start,codes)
                    self.scales_map.publish(start,scales)
            elif data.get_dtype() in ("I8","INT8"):
                scale_key = key.removesuffix(".weight")+".scale"
                with safe_open(root/index[scale_key],framework="pt",device="cpu") as sf:
                    scales = sf.get_slice(scale_key)
                    for start in range(self.start,self.end,chunk_rows):
                        stop = min(start+chunk_rows,self.end)
                        self.codes_map.publish(start,data[start:stop])
                        self.scales_map.publish(start,scales[start:stop])
            else:
                raise ValueError(f"Unsupported Engram dtype {data.get_dtype()}")
        torch.npu.synchronize()
        self.loaded = True
        print(f"ENGRAM_SHARED_LOADED rank={self.query_group.rank} key={key}",flush=True)


@triton.jit
def engram_chunked(codes_ptrs, scales_ptrs, ids, count, valid_mask, out, mask_out,
                    CAPACITY: tl.constexpr, K: tl.constexpr, MASK_SIZE: tl.constexpr,
                    CHUNK: tl.constexpr, WRITE_MASK: tl.constexpr, TILE: tl.constexpr = 1):
    active = tl.load(count)
    col = tl.arange(0,256)
    if TILE == 1:
        for row in range(tl.program_id(0), active, tl.num_programs(0)):
            index = tl.load(ids+row)
            chunk = index//CHUNK
            local = index%CHUNK
            weight = tl.load(codes_ptrs+chunk).to(tl.pointer_type(tl.int8))
            scales = tl.load(scales_ptrs+chunk).to(tl.pointer_type(tl.float32))
            value = tl.load(weight+local*256+col).to(tl.float32)
            scale = tl.load(scales+local*8+col//32)
            tl.store(out+row*256+col,(value*scale).to(tl.bfloat16))
    else:
        # Tiled mode is only for contiguous VMM mappings, not separately
        # registered chunks. Their first pointer addresses the entire table.
        weight = tl.load(codes_ptrs).to(tl.pointer_type(tl.int8))
        scales = tl.load(scales_ptrs).to(tl.pointer_type(tl.float32))
        rr = tl.arange(0,TILE)
        groups = tl.arange(0,8)
        for start in range(tl.program_id(0)*TILE, active, tl.num_programs(0)*TILE):
            row = start+rr
            index = tl.load(ids+row,row<active,0)
            value = tl.load(weight+index[:,None]*256+col[None,:],row[:,None]<active,0).to(tl.float32)
            scale = tl.load(scales+index[:,None]*8+groups[None,:],row[:,None]<active,0)
            result = (value.reshape((TILE,8,32))*scale[:,:,None]).reshape((TILE,256))
            tl.store(out+row[:,None]*256+col[None,:],result.to(tl.bfloat16),row[:,None]<active)
    lane = tl.arange(0,16384)
    for start in range(active*256+tl.program_id(0)*16384,CAPACITY*256,tl.num_programs(0)*16384):
        off = start+lane
        tl.store(out+off,0,off<CAPACITY*256)
    if WRITE_MASK:
        if tl.program_id(0)==0:
            token = tl.arange(0,MASK_SIZE)
            valid = tl.load(valid_mask+token,token*K<active,0)
            tl.store(mask_out+token,valid,token<CAPACITY//K)


class Inputs:
    def __init__(self, tables, layer_ids, capacity, k, outputs=None, mask_output=None, tile=1):
        if tile not in (1,16):
            raise ValueError('Supported lookup tiles are 1 and 16')
        if tile != 1 and any(not hasattr(t.codes_map,'tensor') or not hasattr(t.scales_map,'tensor') for t in tables):
            raise ValueError('Tiled lookup requires contiguous VMM tensor mappings')
        from triton.runtime import driver
        from vllm_ascend.ops.triton.engram_int8 import init_device_properties_triton
        self.tables,self.layers,self.capacity,self.k=tables,layer_ids,capacity,k
        device=tables[0].codes_map.ptrs.device
        self.ids=[torch.empty(capacity*k,dtype=torch.int64,device=device) for _ in tables]
        self.count=torch.zeros(1,dtype=torch.int32,device=device)
        self.valid=torch.empty(capacity,dtype=torch.bool,device=device)
        self.mask=mask_output if mask_output is not None else torch.empty_like(self.valid)
        self.outputs=outputs if outputs is not None else [torch.empty(capacity,k*256,dtype=torch.bfloat16,device=device) for _ in tables]
        init_device_properties_triton()
        self.grid=(driver.active.utils.get_device_properties("npu")["num_vectorcore"],1,1)
        self.kernels=[]
        for i,t in enumerate(tables):
            self.kernels.append(engram_chunked.warmup(t.codes_map.ptrs,t.scales_map.ptrs,self.ids[i],self.count,self.valid,
                self.outputs[i],self.mask,CAPACITY=capacity*k,K=k,MASK_SIZE=triton.next_power_of_2(capacity),
                CHUNK=CHUNK_ROWS,WRITE_MASK=i==0,TILE=tile,grid=self.grid,num_warps=4))

    def prepare(self, hashes, mask):
        n=hashes.shape[0]
        if n>self.capacity: raise ValueError("Engram input exceeds static capacity")
        for i in range(len(self.tables)):
            if n: self.ids[i][:n*self.k].copy_(hashes[:,i].reshape(-1))
        self.count.copy_(torch.tensor([n*self.k],dtype=torch.int32,device="cpu"))
        if n: self.valid[:n].copy_(mask)
        for i,t in enumerate(self.tables):
            self.kernels[i][self.grid](t.codes_map.ptrs,t.scales_map.ptrs,self.ids[i],self.count,self.valid,self.outputs[i],self.mask)
        return {"engram_lookups":dict(zip(self.layers,self.outputs)),"engram_mask":self.mask}
