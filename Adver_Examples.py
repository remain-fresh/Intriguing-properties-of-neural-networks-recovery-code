import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.utils.data as data
from scipy.optimize import minimize
import torch.nn.functional as F
import numpy as np


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
target_label = 1
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

#print(f"从类别 {target_class} ({trainset.classes[target_class]}) 中提取了 {len(selected_dataset)} 个样本"),500个

'''
接下来使用L-BFGS-B算法生成对抗样本。
'''

# 假设 model 是预训练好的分类模型，x 是输入图像（ tensor ）
# 目标是生成对抗样本 x_adv = x + r，让 model(x_adv) 误分类为 target_label

# 1. 修改 objective 函数，增加 x 作为参数
def objective(r_flat, x, model, target_label, c):  # 显式传入所有依赖
    r = r_flat.reshape(x.shape)  # 恢复扰动的形状（与图像一致）
    x_adv = x + r
    # 计算分类损失
    loss = F.cross_entropy(model(x_adv), target_label)
    # 计算扰动幅度惩罚（L2 范数）
    l2_penalty = torch.norm(r, p=2)
    # 总目标
    return (loss + c * l2_penalty).item()

# 2. 同步修改 gradient 函数，增加 x 作为参数
def gradient(r_flat, x, model, target_label, c):
    r = r_flat.reshape(x.shape).requires_grad_(True)
    x_adv = x + r
    loss = F.cross_entropy(model(x_adv), target_label) + c * torch.norm(r, p=2)
    loss.backward()
    return r.grad.flatten().detach().numpy()

def L_BFGS(x,model,target_label,bounds,c):

# 4. 调用优化时，通过 args 传递额外参数
    result = minimize(
        fun=objective,
        x0=r0,
        jac=gradient,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100},
        args=(x, model, target_label, c)  # 将 x 和其他依赖传入目标函数和梯度函数
    )
    return result
cl = 1
cr = 10
r0 = np.zeros((3 * 32 * 32,))
while cl < cr:
    c = (cl + cr) / 2
    print(f"当前 c 值: {c}")
    for i, (x, label) in enumerate(selected_loader):
        x = x.to(device)
        label = label.to(device)
        x_flat = x.flatten()
        bounds = [(-x_flat[i].item(), 1 - x_flat[i].item()) for i in range(x.numel())]
        result = L_BFGS(x,model,target_label,bounds,c)