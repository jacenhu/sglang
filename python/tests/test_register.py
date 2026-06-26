from sglang.srt.models.registry import ModelRegistry

cls, arch = ModelRegistry.resolve_model_cls('Llama3CustomForCausalLM')

print(f'register success: {arch} -> {cls}')
