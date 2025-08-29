import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data

device = 6
evice = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('best_resnet18_cifar10.pth'))
model.to(device)
model.eval()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

'''
接下来抽一个类别中的一定比例样本制作对抗样本。
target_class = 0
target_ratio = 0.1
'''
target_class = 0
target_ratio = 0.1
target_images = []
class_indices = [i for i,(_,label) in enumerate(trainset) if label == target_class]
num_target = int(len(class_indices) * target_ratio)

import random
selected_indices = random.sample(class_indices, num_target)
'''
先自定义一个数据集类，后面作为主要的操作对象。
'''
class SelectedSamplesDataset(data.Dataset):
    def __init__(self, original_dataset, selected_indices):
        self.original_dataset = original_dataset
        self.selected_indices = selected_indices
    
    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, idx):
        return self.original_dataset[self.selected_indices[idx]]

selected_dataset = SelectedSamplesDataset(trainset, selected_indices)
selected_loader = torch.utils.data.DataLoader(selected_dataset, batch_size=1, shuffle=False)

print(f"从类别 {target_class} ({trainset.classes[target_class]}) 中提取了 {len(selected_dataset)} 个样本")
