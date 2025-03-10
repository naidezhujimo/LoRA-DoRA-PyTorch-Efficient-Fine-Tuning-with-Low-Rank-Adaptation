import time
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 划分数据集
BATCH_SIZE = 96
# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.Compose([
                                transforms.RandomRotation(10),  # 随机旋转
                                transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
                                transforms.ToTensor()]),
                               download=True)
test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.Compose([
                                transforms.RandomRotation(10),  # 随机旋转
                                transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
                                transforms.ToTensor()]))

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

#---------------------------------------------------------------------------

# 冻结参数函数
def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False  # 将参数的梯度计算关闭，使其在训练时不更新
        else:
            freeze_linear_layers(child)

# 测试集准确率计算函数
def compute_accuracy(model, data_loader, device=device):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device) 
            targets = targets.to(device)
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float() / num_examples * 100


# 训练函数
def train(num_epochs, model, optimizer, train_loader, device=device):
    start_time = time.time()
    # 定义余弦退火调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs // 2, eta_min=1e-6  # T_max 设置为总 epoch 数，eta_min 设置为最小学习率
    )

    best_accuracy = 0
    patience = 5  # 允许性能不提升的 epoch 数
    counter = 0
    losses = []
    accuracies = []
    max_memory_allocated = 0  # 记录最大显存消耗
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = F.cross_entropy(logits, targets)  # 计算交叉熵损失
            optimizer.zero_grad() 
            loss.backward()  # 更新梯度
            optimizer.step()
        
        # 每个 epoch 结束后更新学习率
        scheduler.step()
        
        # 在验证集上测试性能(早停法)
        val_accuracy = compute_accuracy(model, test_loader)
        # 早停法逻辑
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # 打印当前 epoch 的 loss 和学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}, Val Accuracy: {val_accuracy:.2f}%')
        losses.append(loss.item())
        accuracies.append(compute_accuracy(model, test_loader))
    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    # 打印显存占用
    print(f'Max memory allocated: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 2:.2f} MB')
    # 记录最大显存消耗
    max_memory_allocated = max(max_memory_allocated, torch.cuda.max_memory_allocated(device=device))
        
    return losses, accuracies, max_memory_allocated
#---------------------------------------------------------------------------
# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 7×7 -> 3×3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

# LoRA层实现
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_env = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_env)
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

# DoRA层实现
class LinearWithDoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        self.m = nn.Parameter(self.linear.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        numerator = self.linear.weight + self.lora.alpha * lora.T
        denominator = numerator.norm(p=2, dim=0, keepdim=True)
        directional_component = numerator / denominator
        new_weight = self.m * directional_component
        return F.linear(x, new_weight, self.linear.bias)

# QLoRA层实现
class QLoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, bits=4):
        super().__init__()
        self.bits = bits
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))

    def quantize(self, x):
        # 量化操作
        q_min = -2 ** (self.bits - 1)
        q_max = 2 ** (self.bits - 1) - 1
        scale = (x.max() - x.min()) / (q_max - q_min)
        zero_point = q_min - x.min() / scale
        x_quantized = torch.round(x / scale + zero_point)
        x_quantized = torch.clamp(x_quantized, q_min, q_max)
        return x_quantized, scale, zero_point

    def dequantize(self, x_quantized, scale, zero_point):
        # 反量化操作
        return (x_quantized - zero_point) * scale

    def forward(self, x):
        A_quantized, A_scale, A_zero_point = self.quantize(self.A)
        B_quantized, B_scale, B_zero_point = self.quantize(self.B)
        A_dequantized = self.dequantize(A_quantized, A_scale, A_zero_point)
        B_dequantized = self.dequantize(B_quantized, B_scale, B_zero_point)
        return self.alpha * (x @ A_dequantized @ B_dequantized)

class LinearWithQLoRA(nn.Module):
    def __init__(self, linear, rank, alpha, bits=4):
        super().__init__()
        self.linear = linear
        self.qlora = QLoRALayer(
            linear.in_features, linear.out_features, rank, alpha, bits
        )

    def forward(self, x):
        return self.linear(x) + self.qlora(x)

#---------------------------------------------------------------------------
# 超参数
random_seed = 620
learning_rate = 1e-4
rank = 64
alpha = 8
num_epochs = 50

#---------------------------------------------------------------------------

torch.manual_seed(random_seed)
model = CNN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

model_lora = copy.deepcopy(model)
model_dora = copy.deepcopy(model)
model_qlora = copy.deepcopy(model)

# 将全连接层替换为 LoRA 层
model_lora.fc_layers[0] = LinearWithLoRA(model_lora.fc_layers[0], rank=rank, alpha=alpha)
model_lora.fc_layers[2] = LinearWithLoRA(model_lora.fc_layers[2], rank=rank, alpha=alpha)
freeze_linear_layers(model_lora)
model_lora.to(device)
optimizer_lora = torch.optim.AdamW(model_lora.parameters(), lr=learning_rate, weight_decay=1e-4)

# 将全连接层替换为 DoRA 层
model_dora.fc_layers[0] = LinearWithDoRA(model_dora.fc_layers[0], rank=rank, alpha=alpha)
model_dora.fc_layers[2] = LinearWithDoRA(model_dora.fc_layers[2], rank=rank, alpha=alpha)
freeze_linear_layers(model_dora)
model_dora.to(device)
optimizer_dora = torch.optim.AdamW(model_dora.parameters(), lr=learning_rate, weight_decay=1e-4)

# 将全连接层替换为 QLoRA 层
model_qlora.fc_layers[0] = LinearWithQLoRA(model_qlora.fc_layers[0], rank=rank, alpha=alpha, bits=4)
model_qlora.fc_layers[2] = LinearWithQLoRA(model_qlora.fc_layers[2], rank=rank, alpha=alpha, bits=4)
freeze_linear_layers(model_qlora)
model_qlora.to(device)
optimizer_qlora = torch.optim.AdamW(model_qlora.parameters(), lr=learning_rate, weight_decay=1e-4)


# 不使用任何微调方法
print("Training base model...")
base_losses, base_accuracies, base_memory = train(num_epochs, model, optimizer, train_loader)
print('-------------------------------------')

# 使用 LoRA 方法
print("Training LoRA model...")
lora_losses, lora_accuracies, lora_memory = train(num_epochs, model_lora, optimizer_lora, train_loader)
print('-------------------------------------')

# 使用 DoRA 方法
print("Training DoRA model...")
dora_losses, dora_accuracies, dora_memory = train(num_epochs, model_dora, optimizer_dora, train_loader)
print('-------------------------------------')

# 使用 QLoRA 方法
print("Training QLoRA model...")
qlora_losses, qlora_accuracies, qlora_memory = train(num_epochs, model_qlora, optimizer_qlora, train_loader)
print('-------------------------------------')



plt.figure(figsize=(14, 10))
# 绘制损失值图像
plt.subplot(2, 2, 1)
plt.plot(base_losses, label='Base Model', color='blue')
plt.plot(lora_losses, label='LoRA', color='orange')
plt.plot(dora_losses, label='DoRA', color='green')
plt.plot(qlora_losses, label='QLoRA', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()

# 绘制准确率图像
plt.subplot(2, 2, 2)
plt.plot(base_accuracies, label='Base Model', color='blue')
plt.plot(lora_accuracies, label='LoRA', color='orange')
plt.plot(dora_accuracies, label='DoRA', color='green')
plt.plot(qlora_accuracies, label='QLoRA', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy Comparison')
plt.legend()

# 绘制 Base Model 的损失值和准确率
plt.subplot(2, 2, 3)
plt.plot(base_losses, label='Loss', color='blue')
plt.plot(base_accuracies, label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Base Model: Loss and Accuracy')
plt.legend()

# 绘制 LoRA 的损失值和准确率
plt.subplot(2, 2, 4)
plt.plot(lora_losses, label='Loss', color='blue')
plt.plot(lora_accuracies, label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('LoRA: Loss and Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# 在训练完成后收集准确率
base_accuracy = compute_accuracy(model, test_loader)
lora_accuracy = compute_accuracy(model_lora, test_loader)
dora_accuracy = compute_accuracy(model_dora, test_loader)
qlora_accuracy = compute_accuracy(model_qlora, test_loader)

# 打印最终准确率
print(f'Base Model Test Accuracy: {base_accuracy:.2f}%')
print(f'LoRA Model Test Accuracy: {lora_accuracy:.2f}%')
print(f'DoRA Model Test Accuracy: {dora_accuracy:.2f}%')
print(f'QLoRA Model Test Accuracy: {qlora_accuracy:.2f}%')

# 收集准确率和显存消耗
methods = ['Base Model', 'LoRA', 'DoRA', 'QLoRA']
accuracies = [base_accuracy, lora_accuracy, dora_accuracy, qlora_accuracy]
memory_usage = [base_memory / 1024 ** 2, lora_memory / 1024 ** 2, dora_memory / 1024 ** 2, qlora_memory / 1024 ** 2]  # 转换为 MB

# 绘制柱状图
plt.figure(figsize=(12, 6))
x = np.arange(len(methods))
width = 0.35

# 绘制准确率柱状图
plt.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='blue')
# 绘制显存消耗柱状图
plt.bar(x + width/2, memory_usage, width, label='Memory Usage (MB)', color='orange')

plt.xlabel('Method')
plt.ylabel('Value')
plt.title('Comparison of Test Accuracy and Memory Usage for Different Methods')
plt.xticks(x, methods)
plt.legend()

# 在柱子上方显示数值
for i, (acc, mem) in enumerate(zip(accuracies, memory_usage)):
    plt.text(i - width/2, acc + 1, f'{acc:.2f}%', ha='center', va='bottom')
    plt.text(i + width/2, mem + 1, f'{mem:.2f} MB', ha='center', va='bottom')

plt.show()
