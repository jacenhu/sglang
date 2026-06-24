from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from layers.logits_processor import LogitsProcessorOutput
from model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

from transformers import LlamaConfig

class Llama3CustomModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                forward_batch: ForwardBatch,
                input_embeds: torch.Tensor,
                get_embedding: bool = False,
                pp_proxy_tensors: Optional[PPProxyTensors] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor], PPProxyTensors]]:



class Llama3CustomForCausalLM(nn.Module):
    def __init__(self,
                 config: LlamaConfig) -> None:
        super().__init__()
        self.model = self._init_model(config)

    def _init_model(self,
                    config: LlamaConfig):
        return Llama3CustomModel(config)


    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                forward_batch: ForwardBatch,
                input_embeds: torch.Tensor,
                get_embedding: bool = False,
                pp_proxy_tensors: Optional[PPProxyTensors] = None
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids,
                                   positions,
                                   forward_batch,
                                   input_embeds,
                                   pp_proxy_tensors)
        return hidden_states
