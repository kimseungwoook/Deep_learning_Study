import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# parameter
num_epochs = 10
num_classes = 10
batch_size = 64
learning_rate = 0.001

# 데이터 불러오기
train_dataset = datasets.MNIST(root = './data',
                               train = True,
                               download = True,
                               transform = transforms.ToTensor())

test_dataset = datasets.MNIST(root = './data',
                              train = False,
                              download = True,
                              transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# 데이터 Check
for (x_train, y_train) in train_loader:
  print(f"x_train: {x_train.size()}, type: {x_train.type()}")
  print(f"y_train: {y_train.size()}, type: {y_train.type()}")
  break

# CNN 정의
class ConvNet(nn.Module):
  def __init__(self, num_classes=10):
    super(ConvNet, self).__init__()
    self.layer1 = nn.Sequential(
        # 입력 이미지 크기 : 64 x 1 x 28 x 28
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        # conv 후 크기 = 64 x 16 x14 x 14
        nn.BatchNorm2d(16),
        nn.ReLU(),
        # MaxPooling 후 크기 = 64 x 16 x 7 x 7
        nn.MaxPool2d(kernel_size=2, stride=2))
    
    self.layer2 = nn.Sequential(
        # 입력 크기 : 64 x 16 x 7 x 7
        nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
        # Conv 후 크기 : 64 x 32 x 8 x 8
        nn.BatchNorm2d(32),
        nn.ReLU(),
        # MaxPooling 후 크기 : 64 x 32 x 4 x 4
        nn.MaxPool2d(kernel_size=2, stride=2))
    
    self.fc_layer = nn.Linear(4*4*32, num_classes)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc_layer(out)
    return out


if __name__ == "__main__":
  model = ConvNet(num_classes).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  total_step = len(train_loader)
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      images = images.to(device)
      labels = labels.to(device)

      output = model(images)
      loss = criterion(output, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (i+1) % 100 == 0:
        print(f"Epoch :{epoch+1}/{num_epochs}, Step :{i+1}/{total_step}, Loss: {loss.item():.4f}")

    # 평가 모델
    model.eval()
    with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
      print(f"Test Accuracy: {100*correct/total}%")
