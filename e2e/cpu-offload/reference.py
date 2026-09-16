"""Benchmark adapter: unchanged PR16544/ecb641e CPU lookup/routing code."""
import os
from pathlib import Path
import torch
from pr16544.engram_hbm import NodeShardedEngram

torch.ops.load_library(str(Path(__file__).parent/'pr16544/build/engram_cpu_reference.so'))


class CPUOffloadEngram(NodeShardedEngram):
    def __init__(self,*args,**kwargs):
        if kwargs.get('storage_format')!='int8':
            raise ValueError('This matched baseline requires the original INT8 checkpoint')
        super().__init__(*args,**kwargs,cpu_offload=True)
        assert self.weight.device.type==self.weight_scale.device.type=='cpu'
        assert self.offload_pinned and not self.compressed_int8_wire
        print(f'NATIVE_CPU_OFFLOAD rank={self.query_group.rank} dtype={self.weight.dtype} '
              f'scales={self.weight_scale.dtype} threads={torch.get_num_threads()} '
              f'rows={self.end-self.start} impl=pr16544-ecb641e-fused-cpu',flush=True)


def install(module):
    if os.environ.get('VLLM_ASCEND_ENGRAM_VMM_RUN'):
        raise RuntimeError('CPU baseline must not enable VMM')
    module.NodeShardedEngram=CPUOffloadEngram
