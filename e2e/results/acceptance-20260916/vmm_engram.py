"""Read-only Engram tables in shared host physical memory, mapped at startup."""
import os,json
from pathlib import Path
import torch
from safetensors import safe_open
from vllm_ascend.models.deepseek_v41.engram_hbm import NodeShardedEngram,quantize_engram_rows
from vmm_mapping import VmmMapping
from chunked import Inputs

class VmmEngram(NodeShardedEngram):
    counter=0

    def __init__(self,rows,width,query_group,device=None,storage_format='bf16',**kwargs):
        if storage_format!='int8' or width!=256:
            raise ValueError('Host VMM prototype requires INT8 Engram width256')
        if os.environ.get('ENGRAM_E2E_REUSE')=='1':
            raise ValueError('VMM physical allocations cannot reuse files from a previous process')
        self.control_resident=os.environ.get('ENGRAM_E2E_MODE')=='switch'
        local=torch.device('npu',torch.npu.current_device())
        super().__init__(rows,width,query_group,device=local if self.control_resident else 'meta',storage_format=storage_format)
        index=VmmEngram.counter
        VmmEngram.counter+=1
        root=Path('/dev/shm')/('engram-vmm-'+os.environ['ENGRAM_E2E_RUN'])
        self.codes_map=VmmMapping(root/f'{index}.codes',rows,256,torch.int8,query_group)
        self.scales_map=VmmMapping(root/f'{index}.scales',rows,8,torch.float32,query_group)
        if not self.control_resident:
            self.weight=torch.nn.Parameter(self.codes_map.tensor[self.start:self.end],requires_grad=False)
            self.weight_scale=self.scales_map.tensor[self.start:self.end]
        self.loaded=False
        self.ready_path=root/f'{index}.weights-ready.json'
        self.identity={'backend':'host-vmm','rows':rows,'width':width}

    def load_checkpoint(self,model_path,key,chunk_rows=65536):
        self.identity['checkpoint']=str(Path(model_path).resolve())
        if self.control_resident:
            NodeShardedEngram.load_checkpoint(self,model_path,key,chunk_rows)
            for start in range(self.start,self.end,chunk_rows):
                stop=min(start+chunk_rows,self.end)
                self.codes_map.publish(start,self.weight.data[start-self.start:stop-self.start])
                self.scales_map.publish(start,self.weight_scale[start-self.start:stop-self.start])
        else:
            root=Path(model_path)
            index=json.loads((root/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
            with safe_open(root/index[key],framework='pt',device='cpu') as f:
                source=f.get_slice(key)
                if source.get_shape()!=[self.rows,self.width]: raise ValueError('Engram shape mismatch')
                if source.get_dtype() in ('I8','INT8'):
                    scale_key=key.removesuffix('.weight')+'.scale'
                    with safe_open(root/index[scale_key],framework='pt',device='cpu') as sf:
                        scales=sf.get_slice(scale_key)
                        if scales.get_shape()!=[self.rows,8] or scales.get_dtype()!='F32':
                            raise ValueError('Engram scale format mismatch')
                        for start in range(self.start,self.end,chunk_rows):
                            stop=min(start+chunk_rows,self.end)
                            self.codes_map.publish(start,source[start:stop])
                            self.scales_map.publish(start,scales[start:stop])
                elif source.get_dtype()=='BF16':
                    for start in range(self.start,self.end,chunk_rows):
                        stop=min(start+chunk_rows,self.end)
                        codes,scales=quantize_engram_rows(source[start:stop].to(self.codes_map.device))
                        self.codes_map.publish(start,codes);self.scales_map.publish(start,scales)
                else: raise ValueError('Unsupported Engram checkpoint dtype')
        torch.npu.synchronize()
        self.loaded=True
        print(f'ENGRAM_VMM_LOADED rank={self.query_group.rank} key={key}',flush=True)

class VmmInputs(Inputs):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.initial_counts=self.tables[0].codes_map.lifecycle_counts()
        self.calls=0

    def prepare(self,hashes,mask):
        result=super().prepare(hashes,mask)
        counts=self.tables[0].codes_map.lifecycle_counts()
        if counts!=self.initial_counts:
            raise RuntimeError('Engram VMM mapping lifecycle changed during serving')
        self.calls+=1
        if self.calls%100==1:
            print(f'ENGRAM_VMM_STABLE calls={self.calls} allocations={counts[0]} frees={counts[1]}',flush=True)
        return result
