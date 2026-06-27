from typing import Dict, Iterable, Optional, Union, Tuple, List

import torch
import torch.nn as nn
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from transformers import LlamaConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.logits_processor import LogitsProcessorOutput, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.distributed import get_pp_group
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix


class Llama3CustomAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        # TP分片: QKVParallelLinear需要total heads, 但split尺寸和RadixAttention需要per-TP
        tp_size = get_parallel().tp_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
            self.num_kv_heads = self.total_num_kv_heads // tp_size
        else:
            assert tp_size % self.total_num_kv_heads == 0
            self.num_kv_heads = 1

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.total_num_heads
        )
        self.scaling = self.head_dim ** -0.5
        # per-TP 的 Q/K/V 维度 (与 QKVParallelLinear 实际输出一致)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # QKV融合投影层 (传入 total heads, 内部按TP自动切分输出)
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        # 输出投影层
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        # 旋转位置编码
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", 10000)
            rope_scaling = rope_parameters
        else:
            rope_theta = getattr(config, "rope_theta", 10000)
            rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=getattr(config, "rope_is_neox_style", True),
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(self, hidden_states, positions, forward_batch):
        # 1. 隐藏状态投影为Q、K、V
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 2. 对Q、K施加旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        # 3. 注意力计算
        attn_output = self.attn(q, k, v, forward_batch)
        # 4. 输出投影
        output, _ = self.o_proj(attn_output)
        return output


class Llama3CustomMLP(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )

        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

        self.act_fn = SiluAndMul()

    def forward(self, x, forward_batch = None):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

class Llama3CustomDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # 1. 初始化注意力层
        self.self_attn = Llama3CustomAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        # 2. Llama前馈网络SwiGLU
        self.mlp = Llama3CustomMLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        # 3. 前后2次RMS归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, positions, forward_batch):
        # 前置归一化 -> 自注意力 + 残差连接
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, forward_batch)
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
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """
        初始化Llama3CustomForCausalLM模型的所有子模块。

        Args:
            config: LlamaConfig，模型超参数配置，包含词表大小、隐藏层维度、
                    Transformer层数、RMSNorm epsilon等。
            quant_config: 量化配置。
            prefix: 权重名前缀。
        """
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        # 词嵌入层
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        # Transformer layers
        self.layers = nn.ModuleList([
            Llama3CustomDecoderLayer(
                config=config,
                layer_id=i,
                quant_config=quant_config,
                prefix=add_prefix("model.layers", prefix),
            )
            for i in range(config.num_hidden_layers)
        ])
        # 归一化和线性变换
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        # 权重名称映射：checkpoint中的分片名 -> 模型中的融合参数名
        self.stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                forward_batch: ForwardBatch,
                input_embeds: torch.Tensor = None,
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
        logits = self.logits_processor(input_ids, hidden_stats, self.lm_head, forward_batch)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        权重加载:
        将checkpoint中的分片参数(q_proj, k_proj, v_proj)映射到模型中的融合参数(qkv_proj)。
        同理 gate_proj + up_proj → gate_up_proj。
        """
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "rotary_emb" in name:
                continue
            # 跳过 tie_word_embeddings 情况下的 lm_head
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # 处理checkpoint分片名 -> 模型融合参数名的映射
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # 不需要映射的直接加载
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [
    Llama3CustomForCausalLM,
]
