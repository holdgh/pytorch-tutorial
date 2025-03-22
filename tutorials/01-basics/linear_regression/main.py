import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 超参数 手动设置的参数，用于控制模型的行为和性能
# Hyper-parameters
# 输入层尺寸
input_size = 1
# 输出层尺寸
output_size = 1
# 全量样本进行正向反向的总次数
num_epochs = 60
# 学习率，又名步长，用于权重调整
learning_rate = 0.001

# Toy dataset
# 加载数据集
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Linear regression model
# 选择模型并设定模型参数，输入尺寸和输出尺寸
model = nn.Linear(input_size, output_size)

# Loss and optimizer
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
# 分迭代训练，一次迭代的定义：所有样本进行一次正向和一次反向
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    # 将输入数据由numpy数组转化为torch张量，为了使用torch框架处理计算
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    # 前向处理，将输入数据传入模型，得出输出
    outputs = model(inputs)
    # 计算误差，将模型输出和目标标签输入损失函数，得出误差
    loss = criterion(outputs, targets)

    # Backward and optimize
    # 优化器进行反向传播，梯度下降算法，调整权重参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Plot the graph
# 绘图，两条曲线，训练数据【输入，目标】和预测数据【输入，模型预测输出】
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
