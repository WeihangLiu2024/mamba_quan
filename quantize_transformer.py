from transformers import GPTNeoXForCausalLM, AutoTokenizer
from quantization.Qparam import *
from safetensors.torch import save_file

# 定义模型和 tokenizer 的路径
model_name_or_path = "./pre_trained_model/pythia-2_8b"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 加载模型
model = GPTNeoXForCausalLM.from_pretrained(model_name_or_path)

# 获取模型的所有参数名和对应的维度，并统计总参数数量
params = {}
param_dims = {}
total_params = 0
for name, param in model.named_parameters():
    params[name] = param
    param_dims[name] = tuple(param.size())
    total_params += param.numel()

# fake quantize params
quan_params = {}
for name, param in params.items():
    if "norm" in name:
        qother = QParamOther()
        qother.min_max(param.data)
        quan_params[name] = qother.fake_quantize(param.data)
    else:
        qw = QParamWeight()
        qw.min_max(param.data)
        quan_params[name] = qw.fake_quantize(param.data)

# update & save quantize params
quan_path = "./pre_trained_model/pythia-2_8b-ptq"
for name, param in model.named_parameters():
    param.data = quan_params[name]

model.half()

model.save_pretrained(quan_path, max_shard_size="50GB")
