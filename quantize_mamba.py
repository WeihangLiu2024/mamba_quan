from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from quantization.Qparam import *

# 指定模型和配置文件的路径
path = "/home/liuweihang/workspace/mamba/pre_trained_model/mamba2-2_7b"

# # 加载配置
# config = AutoConfig.from_pretrained(path)

# 加载模型
# model = MambaLMHeadModel(config)
model = MambaLMHeadModel.from_pretrained(path)

# 获取模型的所有参数名和对应的维度，并统计总参数数量
params = {}
param_dims = {}
total_params = 0
for name, param in model.named_parameters():
    params[name] = param
    param_dims[name] = tuple(param.size())
    total_params += param.numel()

# backup
# backup_path = "/home/liuweihang/workspace/mamba/pre_trained_model/mamba2-2_7b_backup"
# model.save_pretrained(backup_path)

# fake quantize params
quan_params = {}
for name, param in params.items():
    if ("dt_bias" in name) or ("A_log" in name) or ("norm" in name) or ("D" in name):
        qother = QParamOther()
        qother.min_max(param.data)
        quan_params[name] = qother.fake_quantize(param.data)
    # elif ("norm" in name) or ("D" in name):
    #     quan_params[name] = param.data
    else:
        qw = QParamWeight()
        qw.min_max(param.data)
        quan_params[name] = qw.fake_quantize(param.data)

# update & save quantize params
quan_path = "/home/liuweihang/workspace/mamba/pre_trained_model/mamba2-2_7b_ptq"
for name, param in model.named_parameters():
    param.data = quan_params[name]
model.save_pretrained(quan_path)



# # 打印参数名和维度
# for name, dim in param_dims.items():
#     print(f"{name}: {dim}")
#
# # 打印总参数数量
# print(f"Total number of parameters: {total_params}")

# # fake quantize to the model
#
#
# # 指定保存的文件夹
# save_directory = "path/to/save/directory"
#
# # 保存修改后的模型和配置
# model.save_pretrained(save_directory)
# config.save_pretrained(save_directory)
