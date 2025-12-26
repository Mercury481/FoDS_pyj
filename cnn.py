import torch
torch.cuda.empty_cache() # 清理残留显存

import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# 复现随机结果用
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

# 0. 设备配置：自动检测是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

# 1 超参数设置
EPOCH = 7                
BATCH_SIZE = 50
LR = 0.001            
DOWNLOAD_MNIST = True

# 2 Mnist 数据集准备
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

# 2.1 数据增强
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                  
    transform=torchvision.transforms.Compose([                    # 数据增强
    torchvision.transforms.RandomRotation(15),                    # 随机旋转正负 15 度
    torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)), # 随机平移
    torchvision.transforms.ToTensor(),
]),
    download=DOWNLOAD_MNIST,
)

# 2.2 数据加载器
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 2.3 选择测试样本 
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000].to(device) / 255.
test_y = test_data.targets[:2000].to(device)

# 3 卷积神经网络设置
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 14x14
            nn.Dropout(0.25)                   # 防止过拟合
        )
        # 第二层卷积：32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 7x7
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)               # 将 conv2 的输出层展平为 (batch_size, 32 * 7 * 7)
        output = self.fc(x)
        return output, x

cnn = CNN().to(device)
print(cnn)

# 3.1 优化器与损失函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)     # 学习率衰减
scheduler.step()  

# 4 训练与测试

best_accuracy = 0.0  # 用于记录历史最高精度
last_epoch_accuracies = []  # 用于存储最后一个 Epoch 的所有测试点精度

for epoch in range(EPOCH):
    cnn.train()
    for step, (b_x, b_y) in enumerate(train_loader):   
        b_x, b_y = b_x.to(device), b_y.to(device)

        output = cnn(b_x)[0]            
        loss = loss_func(output, b_y)   
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()                

        if step % 50 == 0:
            cnn.eval()
            with torch.no_grad():
                test_output, _ = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
                target_y = test_y.cpu().data.numpy()  
                # 计算准确率
                accuracy = float((pred_y == target_y).sum()) / float(target_y.size)
            print(f'Epoch: {epoch+1} | Step: {step} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.5f}')
            
            # 自动保存表现最好的模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(cnn.state_dict(), 'best_mnist_model.pth')
                print(f'---> 发现更优模型！已保存，当前最高准确率: {best_accuracy:.5f}')
            # 记录最后一个 Epoch 的所有精度点
            if epoch == EPOCH - 1:
                last_epoch_accuracies.append(accuracy)
            cnn.train()

    # 每个 Epoch 结束更新学习率
    scheduler.step()

# 计算并打印最后一个 Epoch 的平均精度
if last_epoch_accuracies:
    avg_acc_last_epoch = sum(last_epoch_accuracies) / len(last_epoch_accuracies)
    print('-' * 55)
    print(f'训练结束！')
    print(f'历史最高测试准确率 (Best): {best_accuracy:.5f}')
    print(f'最后一个 Epoch 平均准确率 (Avg): {avg_acc_last_epoch:.5f}')
    print(f'最佳模型文件: best_mnist_model.pth')