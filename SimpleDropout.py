import torch

class CustomDropout:
    def __init__(self, p=0.5):
        self.p = p  # 丢弃概率

    def forward(self, x):
        if self.training:  # 仅在训练时应用 Dropout
            # 生成一个与输入相同形状的随机张量
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)  # 应用丢弃并调整输出
        return x  # 测试时返回原始输入

# 使用示例
dropout = CustomDropout(p=0.5)
dropout.training = True  # 设置为训练模式

# 创建一个示例输入张量
input_tensor = torch.randn(3, 5)  # 形状为 (3, 5)

print(input_tensor)

# 应用自定义 Dropout
output_tensor = dropout.forward(input_tensor)
print(output_tensor)
