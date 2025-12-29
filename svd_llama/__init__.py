from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from .configuration_asvd_llama import ASVDLlamaConfig
from .modeling_asvd_llama import ASVDLlamaForCausalLM

AutoConfig.register("svdllama", ASVDLlamaConfig)
AutoModelForCausalLM.register(ASVDLlamaConfig, ASVDLlamaForCausalLM)
AutoTokenizer.register(ASVDLlamaConfig, LlamaTokenizer)