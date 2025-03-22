import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
# 获取当前运行环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# 超参数设置
# 输入尺寸
input_size = 784
# 隐藏层尺寸
hidden_size = 500
# 类别数目【0-9的10个数字】
num_classes = 10
# 全量数据训练【前向和反向】次数
num_epochs = 5
# 全量数据的分批训练，每批数据的数目
batch_size = 100
# 学习率，步长
learning_rate = 0.001

# MNIST dataset
# 下载数字识别数据集
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
# 加载下载好的数据集，此处已经进行了批量处理
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Fully connected neural network with one hidden layer
# 仅有一个隐藏层的全连接神经网络
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # 初始化神经网络
        super(NeuralNet, self).__init__()
        # 输入层和隐藏层关系设置
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 激活函数 relu
        self.relu = nn.ReLU()
        # 隐藏层和输出层关系设置
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        前向传播关系
        """
        # 输入层到隐藏层，线性组合关系
        out = self.fc1(x)
        # 隐藏层经过激活函数处理，再经线性组合，映射到输出层
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 将模型【神经网络、输入尺寸、隐藏层尺寸、类别数目【输出尺寸】】写入cpu设备
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
# 设置损失函数和优化器【需要模型权重参数和学习率参数】
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    # 分批进行训练，每批100张图片，训练数据共60000张，600批
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        # 将图片矩阵重塑为28*28，写入cpu设备
        images = images.reshape(-1, 28 * 28).to(device)
        # 将标签写入cpu设备
        labels = labels.to(device)

        # Forward pass
        # 依据网络模型前向传播
        outputs = model(images)
        # 计算损失【误差】【outputs为100*10的矩阵，labels为100*1的矩阵，二者是如何计算误差的？】
        loss = criterion(outputs, labels)

        # Backward and optimize
        # 反向传播，梯度下降方向，依据学习率调整权重参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每100张，打印训练信息【哪次迭代，哪一步，损失值多少】
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# 测试模型，在测试阶段，不需要计算梯度【no_grad】
with torch.no_grad():
    # 初始化正确数量
    correct = 0
    # 初始化测试数据总量
    total = 0
    for images, labels in test_loader:
        # 将测试图片按照与训练阶段同样的重塑方式写入cpu设备
        images = images.reshape(-1, 28 * 28).to(device)
        # 将测试图片的标签写入cpu设备
        labels = labels.to(device)
        # 将测试图片放入训练好的模型计算输出值
        outputs = model(images)
        # 输出值为100*10的矩阵，在每一行【dim=1】中取最大值输出【该行的最大值，该最大值在该行的索引】，预测值predicted为一张图片输出值行向量最大元素所在的索引
        _, predicted = torch.max(outputs.data, dim=1)
        # 测试数据总量加1
        total += labels.size(0)
        # 当预测值和标签一致时，测试正确数量加1
        correct += (predicted == labels).sum().item()
    # 测试精度采取测试正确数量比上测试数据总量的百分比形式衡量
    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Save the model checkpoint
# 保存训练好的模型
torch.save(model.state_dict(), 'model.ckpt')
