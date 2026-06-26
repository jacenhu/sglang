"""验证 llama3_custom 模型注册和语法正确性。

注意：SGLang 完整初始化需要 GPU（CUDA/ROCm）。
WSL 无 GPU 环境下只能验证注册和语法，无法跑完整 forward。
在有 GPU 的机器上取消下面的注释来完整测试。
"""
import torch
from transformers import LlamaConfig
from sglang.srt.models.registry import ModelRegistry

# ---- 第一步：验证模型注册 ----
cls, arch = ModelRegistry.resolve_model_cls('Llama3CustomForCausalLM')
print(f'✅ 注册成功: {arch} -> {cls.__name__}')

# 验证继承关系
import inspect
print(f'   基类: {[c.__name__ for c in cls.__mro__[:3]]}')

# ---- 第二步：验证 forward 签名（语法检查） ----
sig = inspect.signature(cls.forward)
print(f'   forward 参数: {list(sig.parameters.keys())}')

# ---- 第三步：简单配置检查 ----
config = LlamaConfig(
    vocab_size=32000,
    hidden_size=128,
    intermediate_size=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    rms_norm_eps=1e-6,
)
config.architectures = ['Llama3CustomForCausalLM']
print(f'✅ LlamaConfig 创建成功 (hidden={config.hidden_size}, layers={config.num_hidden_layers})')

print('\n===== 基本验证全部通过 =====')
print('实例化测试需要在有 GPU 的机器上运行。')

# ---- 以下在 GPU 机器上取消注释 ----
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel

server_args = ServerArgs(model_path="/path/to/model", disable_cuda_graph=True)
set_global_server_args_for_scheduler(server_args)
init_distributed_environment(world_size=1, rank=0,
    distributed_init_method="tcp://127.0.0.1:23456", local_rank=0, backend="nccl")
initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

model = cls(config)
print(f'init success, parameter quantity: {sum(p.numel() for p in model.parameters()):,}')