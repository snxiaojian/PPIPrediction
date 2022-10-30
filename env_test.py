import torch

print(torch.cuda.current_device())

# 返回gpu数量
print(torch.cuda.device_count())

# 返回gpu名称，索引从0开始
print(torch.cuda.get_device_name(0))

# cuda是否可用
print(torch.cuda.is_available())

print(torch.version.cuda)

