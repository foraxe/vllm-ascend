"""Build only PR16544's unchanged CPU operator, not all Ascend custom ops."""
from pathlib import Path
from torch.utils.cpp_extension import load

root=Path(__file__).parent/'pr16544'
(root/'build').mkdir(exist_ok=True)
print(load(name='engram_cpu_reference',sources=[str(root/'engram_cpu.cpp')],
    extra_cflags=['-O3','-fopenmp'],extra_ldflags=['-fopenmp'],
    build_directory=str(root/'build'),is_python_module=False,verbose=True))
