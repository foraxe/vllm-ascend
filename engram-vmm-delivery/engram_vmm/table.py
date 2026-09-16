"""Read-only Engram tables in shared host physical memory, mapped at startup."""
import os,json,re
from pathlib import Path
import torch
from safetensors import safe_open
from vllm_ascend.models.deepseek_v41.engram_hbm import NodeShardedEngram,quantize_engram_rows
from .mapping import VmmMapping

class VmmEngram(NodeShardedEngram):
    counter=0

    def __init__(self,rows,width,query_group,device=None,storage_format='bf16',**kwargs):
        if storage_format!='int8' or width!=256:
            raise ValueError('Host VMM prototype requires INT8 Engram width256')
        run=os.environ['VLLM_ASCEND_ENGRAM_VMM_RUN']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',run):
            raise ValueError('VMM run ID must be a fresh, safe name shared by all ranks')
        super().__init__(rows,width,query_group,device='meta',storage_format=storage_format,**kwargs)
        index=VmmEngram.counter
        VmmEngram.counter+=1
        root=Path('/dev/shm')/('engram-vmm-'+run)
        self.closed=False
        self.codes_map=VmmMapping(root/f'{index}.codes',rows,256,torch.int8,query_group)
        try:
            self.scales_map=VmmMapping(root/f'{index}.scales',rows,8,torch.float32,query_group)
        except Exception as exc:
            self.codes_map._rollback(exc)
            raise
        self.weight=torch.nn.Parameter(self.codes_map.tensor[self.start:self.end],requires_grad=False)
        self.weight_scale=self.scales_map.tensor[self.start:self.end]
        self.loaded=False

    def load_checkpoint(self,model_path,key,chunk_rows=65536):
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


    def close(self):
        # Stop model/graph users first. Mapping.close also checks tensor aliases.
        self.closed=True
        self.weight=None
        self.weight_scale=None
        errors=[]
        for mapping in (self.scales_map,self.codes_map):
            try:
                mapping.close()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('; '.join(errors))
