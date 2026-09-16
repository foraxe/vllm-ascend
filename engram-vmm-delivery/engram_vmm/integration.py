"""Opt-in direct VMM path; absent opt-in leaves the original HBM model untouched."""
import torch
import torch.distributed as dist
import weakref
from vllm.forward_context import get_forward_context


_models=weakref.WeakSet()


def close_model(model):
    """Called only after inference/graph execution has stopped."""
    inputs=getattr(model,'_direct_inputs',None)
    if inputs is not None:
        inputs.close()
    errors=[]
    for layer in model.layers:
        table=getattr(getattr(layer,'engram',None),'embed',None)
        if table is not None and hasattr(table,'close'):
            try:
                table.close()
            except Exception as exc:
                errors.append(str(exc))
    if errors:
        raise RuntimeError('Engram cleanup: '+'; '.join(errors))
    _models.discard(model)
    print('ENGRAM_VMM_MODEL_CLOSED',flush=True)


def install(module):
    from .table import VmmEngram as SharedEngram
    from .lookup import Inputs
    if getattr(module, '_engram_vmm_installed', False): return
    module._engram_vmm_installed=True
    module.NodeShardedEngram = SharedEngram
    original_load = module.AscendDeepseekV41ForCausalLM.load_weights

    def load_weights(self, weights):
        try:
            result = original_load(self, weights)
            tables=[self.model.layers[i].engram.embed for i in self.model.config.engram_layer_ids]
            if not all(t.loaded for t in tables):
                raise RuntimeError("Engram loader did not populate every table")
            torch.npu.synchronize()
            dist.barrier(group=tables[0].query_group.cpu_group)
            _models.add(self.model)
            print("ENGRAM_VMM_MODEL_READY",flush=True)
            return result
        except Exception:
            close_model(self.model)
            raise

    def prepare_inputs(self,input_ids,positions,padded_tokens=None):
        c=self.config
        k=(c.engram_max_ngram_size-1)*c.engram_n_heads
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
    def forbidden_routed_fallback(self,*args,**kwargs):
        raise RuntimeError('VMM forward must receive refreshed lookup inputs')
    module.DeepseekV41Model.prepare_engram=forbidden_routed_fallback
    from vllm_ascend.worker.worker import NPUWorker
    original_shutdown=NPUWorker.shutdown
    def shutdown(worker):
        models=list(_models)
        try:
            original_shutdown(worker)
        finally:
            errors=[]
            for model in models:
                try:
                    close_model(model)
                except Exception as exc:
                    errors.append(str(exc))
            from .mapping import library
            lib=library()
            if lib.host_vmm_retry_rollbacks():
                errors.append(lib.host_shared_last_error().decode())
            if errors:
                raise RuntimeError('; '.join(errors))
    NPUWorker.shutdown=shutdown
