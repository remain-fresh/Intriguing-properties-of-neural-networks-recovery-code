import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
<<<<<<< HEAD
<<<<<<< HEAD
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
=======
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
>>>>>>> 3b644c5c331e02cb9e7e982bf788117a8bc20ff8
=======
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
>>>>>>> 3b644c5c331e02cb9e7e982bf788117a8bc20ff8
print(f"使用设备: {device}")

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# 获取未预训练的ResNet-18模型
model = resnet18(pretrained=False)  # pretrained=False表示不加载预训练权重

# 调整ResNet-18以适应CIFAR-10
# 1. 调整第一个卷积层以适应32x32输入
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# 2. 移除第一个最大池化层，因为输入已经很小
model.maxpool = nn.Identity()
# 3. 调整最后一个全连接层以适应10个类别
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# 训练模型
def train(epochs=100):
    best_acc = 0.0
    # 添加训练指标记录列表
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_loss = 0.0  # 用于计算整个epoch的平均损失
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            running_loss += loss.item()
            if i % 100 == 99:    # 每100个批次打印一次信息
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # 计算并记录平均训练损失
        epoch_loss = total_loss / len(trainloader)
        train_losses.append(epoch_loss)
        
        # 测试模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f'Epoch {epoch+1} 测试准确率: {acc:.2f}%')
        test_accuracies.append(acc)  # 记录测试准确率
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_resnet18_cifar10.pth')
        
        scheduler.step()
    
    print('训练完成')
    print(f'最佳测试准确率: {best_acc:.2f}%')
    
    # 绘制训练指标图表
    plt.figure(figsize=(12, 5))
    # 第一个子图：训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.title('训练损失曲线')
    plt.legend()
    
    # 第二个子图：测试准确率
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.title('测试准确率曲线')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')  # 保存图表
    plt.show()

# 开始训练
HEAD
train(epochs=35)
