from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead

from transformers import LlamaConfig

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors


class Llama3CustomAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads

        self.atten = RadixAttention(
            num_heads = config.num_heads,
            head_dim =
            scaling =
            num_kv_heads = config.num_key_value_heads,
            layer_id = layer_id
        )



class Llama3CustomDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_id: int):
        super().__init__()
        #1. 初始化注意力层
        self.self_atten = Llama3CustomAttention(
            config,
            layer_id
        )

        # 2. Llama前馈网络SwiGLU
        self.mlp = SiluAndMul(
            hidden_size = config.hidden_size,
            intermediate_size = config.intermediate_size
        )

        # 3. 前后2次RMS归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def forward(self, hidden_states, positions, forward_batch):
        # 前置归一化 -> 自注意力 + 残差连接
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_atten(hidden_states, positions, forward_batch)
        hidden_states += residual

        # 注意力输出后归一化 -> 前馈网络 + 残差连接
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual

        return hidden_states


class Llama3CustomForCausalLM(nn.Module):
    """
    基于Llama3架构的自定义因果语言模型。
    整合词嵌入、多层Transformer、归一化及语言模型头，用于自回归文本生成。
    """
    def __init__(self,
                 config: LlamaConfig) -> None:
        """
        初始化Llama3CustomForCausalLM模型的所有子模块。

        Args:
            config: LlamaConfig，模型超参数配置，包含词表大小、隐藏层维度、
                    Transformer层数、RMSNorm epsilon等。
        """
        super().__init__()
        self.config = config
        # 词嵌入层
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # Transformer layers
        self.layers = nn.ModuleList([
            Llama3CustomDecoderLayer(config, layer_id = i)
            for i in range(config.num_hidden_layers)
        ])
        # 归一化和线性变换
        self.norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                forward_batch: ForwardBatch,
                input_embeds: torch.Tensor,
                get_embedding: bool = False,
                pp_proxy_tensors: Optional[PPProxyTensors] = None
    ) -> LogitsProcessorOutput:
        # 1. token编码为向量
        hidden_stats = self.embed_tokens(input_ids)
        # 2. 逐层向前传播
        for layer in self.layers:
            hidden_stats = layer(hidden_stats, positions, forward_batch)
        # 3. 最后归一化，输出每个token的预测概率
        hidden_stats = self.norm(hidden_stats)
        logits = self.logits_processor(input_ids, hidden_stats, self.lm_head)
        return logits

    def load_weights(self, weights):
        """
        权重加载:
        Checkpoint = 训练好的模型文件包，包含配置（怎么搭模型）和权重（模型参数数值）。
        sglang 的工作就是把这个包读进来，重建模型结构，填入权重数据，然后执行推理。你在
        load_weights里看到的所有逻辑（名字映射、融合参数、PP过滤等），本质上就是在处理checkpoint中的张量名和模型参数名之间的差异。

        在权重加载场景中必须用 named_parameters()，因为需要靠名字来匹配 checkpoint 中的权重和模型中的参数。
        """
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb" in name:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = [
    Llama3CustomForCausalLM,
]
