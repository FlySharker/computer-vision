import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

# 定义批次大小、学习率、动量和周期数
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

# 定义数据转换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义卷积神经网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(5, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(160, 64),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


# 实例化模型
model = Net()

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# 训练模型函数
def mnist_train():
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    Acc = 0.0
    Loss = 0.0
    acc_list_train = []
    loss_list_train = []
    for epoch in range(EPOCH):
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            running_total += inputs.shape[0]
            running_correct += (predicted == target).sum().item()
            acc = 100 * running_correct / running_total

            if batch_idx % 100 == 99:
                print('epoch:%d, batch_idx:%d, loss: %.3f , acc: %.2f %%'
                      % (epoch + 1, batch_idx + 1, running_loss / 100, acc))
                running_loss = 0.0
                running_total = 0
                running_correct = 0
        Acc = acc
        Loss = running_loss / 100
        acc_list_train.append(Acc)
        loss_list_train.append(Loss)
    return acc_list_train, loss_list_train


# 测试模型函数
def mnist_test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('Accuracy on test set: %.1f %% ' % (100 * acc))


# 主函数
if __name__ == '__main__':
    acc_list = []
    loss_list = []
    acc_list, loss_list = mnist_train()

    # 绘制准确率图像
    plt.plot(acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On Training-Set')
    plt.show()

    # 绘制损失图像
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss On Training-Set')
    plt.show()

    # 测试模型
    mnist_test()
