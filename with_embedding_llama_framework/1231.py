import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 输入维度为2，输出维度为1

    def forward(self, x):
        output = self.linear(x)
        return output

# 初始化模型
model = LinearModel()

# 定义额外的参数
extra_param = nn.Parameter(torch.tensor([0.5]))  # 额外的参数

# 定义优化器，并将额外参数加入到优化器中
optimizer = optim.SGD([{'params': model.parameters()}, {'params': [extra_param]}], lr=0.1)

# 输入数据
x = torch.tensor([[1.0, 2.0]])

# 计算输出
output = model(x)

# 将额外参数与模型输出相加
output = output + extra_param

# 计算损失
loss = torch.mean(output)

# 反向传播和参数更新
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 查看更新后的额外参数值
print(extra_param)
