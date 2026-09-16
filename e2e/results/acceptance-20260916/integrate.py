"""Opt-in model patch installed only in the isolated old-image E2E pods."""
import torch
import torch.distributed as dist
import os
import json
from pathlib import Path
from vllm.forward_context import get_forward_context


def install(module):
    from chunked import SharedEngram, Inputs
    if os.environ.get('ENGRAM_E2E_BACKEND')=='vmm':
        from vmm_engram import VmmEngram as SharedEngram, VmmInputs as Inputs
    elif os.environ.get("ENGRAM_E2E_BOUNDED") == "1":
        from bounded import BoundedInputs as Inputs
    module.NodeShardedEngram = SharedEngram
    original_load = module.AscendDeepseekV41ForCausalLM.load_weights
    original_prepare = module.DeepseekV41Model.prepare_engram_inputs

    def load_weights(self, weights):
        result = original_load(self, weights)
        tables=[self.model.layers[i].engram.embed for i in self.model.config.engram_layer_ids]
        assert all(t.loaded for t in tables), "Engram loader did not populate every table"
        torch.npu.synchronize()
        dist.barrier(group=tables[0].query_group.cpu_group)
        if tables[0].query_group.rank==0:
            for table in tables:
                table.ready_path.write_text(json.dumps(table.identity,sort_keys=True)+'\n')
        dist.barrier(group=tables[0].query_group.cpu_group)
        print("ENGRAM_DIRECT_MODEL_READY",flush=True)
        return result

    def prepare_inputs(self,input_ids,positions,padded_tokens=None):
        switched = os.environ.get("ENGRAM_E2E_MODE") == "switch"
        direct = not switched or Path("/dev/shm/engram-direct-enabled").exists()
        c=self.config
        k=(c.engram_max_ngram_size-1)*c.engram_n_heads
        if not direct:
            result=original_prepare(self,input_ids,positions,padded_tokens)
            if not hasattr(self,"_direct_inputs"):
                tables=[self.layers[i].engram.embed for i in c.engram_layer_ids]
                outputs=[self._engram_input_buffers[0][i] for i in c.engram_layer_ids]
                self._direct_inputs=Inputs(tables,c.engram_layer_ids,self._engram_max_tokens,k,
                                           outputs=outputs,mask_output=self._engram_input_buffers[1])
            if hasattr(self._direct_inputs,'previous_rows'):
                self._direct_inputs.previous_rows=self._engram_max_tokens*k
            return result
        ctx=get_forward_context()
        if getattr(ctx,"flash_comm_v1_enabled",False):
            raise RuntimeError("First direct E2E variant requires FlashComm1 disabled")
        hashes=torch.empty((0,len(c.engram_layer_ids),k),dtype=torch.int64,device="cpu")
        mask=torch.empty(0,dtype=torch.bool,device="cpu")
        metadata=ctx.attn_metadata
        if metadata is not None and self.engram_history is not None:
            first=self.layers[0].self_attn.dsa_attn.swa_cache_layer
            meta=metadata[first.prefix]
            boundaries=meta.query_start_loc_cpu if getattr(meta,"query_start_loc_cpu",None) is not None else meta.query_start_loc.detach().cpu()
            boundaries=boundaries.long()
            n=int(boundaries[-1])
            requests=torch.repeat_interleave(torch.arange(len(boundaries)-1,device="cpu"),boundaries.diff())
            blocks=meta.block_table_cpu if getattr(meta,"block_table_cpu",None) is not None else meta.block_table.detach().cpu()
            hashes,mask=self.engram_history.update(input_ids[:n].cpu().long(),positions[:n].cpu().long(),requests,blocks,meta.storage_block_size)
        if not hasattr(self,"_direct_inputs"):
            tables=[self.layers[i].engram.embed for i in c.engram_layer_ids]
            self._direct_inputs=Inputs(tables,c.engram_layer_ids,self._engram_max_tokens,k)
        self._engram_direct_calls=getattr(self,"_engram_direct_calls",0)+1
        if self._engram_direct_calls==1: print("ENGRAM_DIRECT_PREPARE_ACTIVE",flush=True)
        return self._direct_inputs.prepare(hashes,mask)

    module.AscendDeepseekV41ForCausalLM.load_weights=load_weights
    module.DeepseekV41Model.prepare_engram_inputs=prepare_inputs
    if os.environ.get('ENGRAM_E2E_MODE')=='direct':
        def forbidden_routed_fallback(self,*args,**kwargs):
            raise RuntimeError('Direct Engram forward must receive refreshed lookup inputs; routed fallback forbidden')
        module.DeepseekV41Model.prepare_engram=forbidden_routed_fallback
