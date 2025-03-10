import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # 计算标准差
        std_env = 1 / torch.sqrt(torch.tensor(rank).float())
        # 将输入特征从维度 in_dim 映射到低秩空间 rank
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_env)
        # 将低秩空间 rank 的特征进一步映射到目标维度 out_dim
        self.B = nn.Parameter(torch.zeros(rank, out_dim)) 
        self.alpha = alpha
    
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
torch.manual_seed(620)
layer = nn.Linear(10, 2)
x = torch.randn((1, 10))
layer_lora_1 = LinearWithLoRA(layer, rank=2, alpha=4)
print('Oraginal output:', layer(x))
print('LoRA Output:', layer_lora_1(x))

class MLP(nn.Module):
    def __init__(self, in_features, hidden1, hidden2, classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, classes)
        )

    def forward(self, x):
        return self.layers(x)
    
in_features, hidden1, hidden2, classes = 784, 128, 256, 10
model = MLP(in_features, hidden1, hidden2, classes)

print(model)

model.layers[0] = LinearWithLoRA(model.layers[0], rank=4, alpha=8)
model.layers[2] = LinearWithLoRA(model.layers[2], rank=4, alpha=8)
model.layers[4] = LinearWithLoRA(model.layers[4], rank=4, alpha=8)

print(model)

def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            freeze_linear_layers(child)

freeze_linear_layers(model)
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")

#--------------------------------------------------------------------------


class LinearWithDoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        self.m = nn.Parameter(self.linear.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        numerator = self.linear.weight + self.lora.alpha*lora.T
        denominator = numerator.norm(p=2, dim=0, keepdim=True)
        directional_component = numerator / denominator
        new_weight = self.m * directional_component
        return F.linear(x, new_weight, self.linear.bias)
    