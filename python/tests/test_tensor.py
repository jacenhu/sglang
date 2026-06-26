import torch
from transformers import LlamaConfig
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

# 初始化运行时环境（GPU 机器需要）
server_args = ServerArgs(model_path="/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
                         disable_cuda_graph=True)
set_global_server_args_for_scheduler(server_args)
init_distributed_environment(world_size=1, rank=0,
    distributed_init_method="tcp://127.0.0.1:23456", local_rank=0, backend="nccl")
initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

config = LlamaConfig(
    vocab_size=1000, hidden_size=64, intermediate_size=256,
    num_attention_heads=4, num_key_value_heads=2,
    num_hidden_layers=2, rms_norm_eps=1e-5,
    architectures=['Llama3CustomForCausalLM']
)

cls, _ = ModelRegistry.resolve_model_cls('Llama3CustomForCausalLM')
model = cls(config)

# 模拟一次推理
batch_size = 1
hidden_size = config.hidden_size
input_ids = torch.randint(0, 1000, (batch_size, 1))
positions = torch.zeros(batch_size, 1, dtype=torch.long)
forward_batch = ForwardBatch(
    forward_mode=ForwardMode.DECODE,
    batch_size=batch_size,
    input_ids=input_ids,
    req_pool_indices=torch.zeros(batch_size, dtype=torch.int32),
    req_to_token_pool=None,
    out_cache_loc=torch.zeros(batch_size, dtype=torch.int64),
    seq_lens=torch.ones(batch_size, dtype=torch.int32),
    extend_seq_lens=None,
    return_logprob=False,
)

output = model(input_ids, positions, forward_batch, None)
print(f'✅ Forward 成功，输出 shape: {output.logits.shape}')