import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from google.colab import drive

# 구글 드라이브 마운트
drive.mount('/content/drive')
Data_path = '/content/drive/MyDrive/Mnist_dataset'

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"다음 기기로 학습 : {device}")

# 난수 설정
random.seed(777)
torch.manual_seed(777)
if device == "cuda":
  torch.cuda.manual_seed_all(777)

# 파라미터 설정
lr = 0.001
training_epochs = 15
batch_size = 100
drop_prob = 0.3

# dataset 설정
mnist_train = dataset.MNIST(root=Data_path,
                            train=True,
                            download = True,
                            transform = transforms.ToTensor())
mnist_test = dataset.MNIST(root=Data_path,
                           train = False,
                           transform = transforms.ToTensor(),
                           download = True)

# 데이터로더 = 해당 데이터를 배치 사이즈와 섞어서 가져오는 툴
mnist_dataloader = DataLoader(dataset=mnist_train,
                              batch_size = batch_size,
                              shuffle = True,
                              drop_last = True)

# 모델 설계
linear1 = nn.Linear(28*28, 512, bias=True)
linear2 = nn.Linear(512, 512, bias=True)
linear3 = nn.Linear(512, 512, bias=True)
linear4 = nn.Linear(512, 10, bias=True)

relu = nn.ReLU()
dropout = nn.Dropout(p=drop_prob)

# 각 층의 가중치를 초기화
nn.init.xavier_uniform_(linear1.weight)
nn.init.xavier_uniform_(linear2.weight)
nn.init.xavier_uniform_(linear3.weight)
nn.init.xavier_uniform_(linear4.weight)

# 모델 생성
model = nn.Sequential(
    linear1, relu, dropout,
    linear2, relu, dropout,
    linear3, relu, dropout,
    linear4
).to(device)

# 손실 함수와 최적화
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# train 및 backpropagation
total_batch = len(mnist_dataloader)
model.train()

for epoch in range(training_epochs):
  total_cost = 0
  for X, Y in mnist_dataloader:
    # X는 이미지, Y는 라벨링
    # X를 1차원으로 펼쳐서 입력을 해야 함
    # 또한 배치 크기를 100으로 설정하여서 X의 shape = (100,784)이다.
    X = X.view(-1, 28*28).to(device)
    Y = Y.to(device)

    # 역전파 계산을 할 때마다 기울기 값을 누적시키기 때문에
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    prediction = model(X)
    # 예측값과 정답값 비교 = 손실 값
    cost = criterion(prediction, Y)
    # backpropagation
    cost.backward()
    # backpropagation으로 계산된 가중치로 업데이트 되고
    # 다음 epoch으로 넘어가기
    optimizer.step()

    total_cost += cost
  avg_cost = total_cost / total_batch
  print("Epoch :", "%04d" %(epoch+1), 'Cost = ', '{:.9f}'.format(avg_cost))

# 학습 완료 !!
print("Learning finished!!!!")



# 테스트 데이터를 사용해서 모델 테스트

# torch.no_grad()를 하면은 gradient 계산을 수행하지 않음
# 테스트니까 기울기를 구하여 극솟값을 구할 이유가 없음
with torch.no_grad():
  X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
  Y_test = mnist_test.test_labels.to(device)

  prediction = model(X_test)
  # torch.argmax에서 1은 가로방향으로 가장 큰 예측 값 찾기
  # prediction은 batch_size가 100이기 때문에 (100, 10) 크기 일 것임
  # 즉 한 행이 MNIST 숫자 데이터를 예측한 값이라는 거지 따라서 argmax에서 1(가로방향)을 설정한다.
  correct_prediction = torch.argmax(prediction, 1) == Y_test
  accuracy = correct_prediction.float().mean()
  print(f"Accuracy : {accuracy.item()}")

  # mnist_test에서 무작위로 하나를 뽑아서 테스트하여 시각화
  r = random.randint(0, len(mnist_test)-1)
  X_single_data = mnist_test.test_data[r:r+1].view(-1, 28*28).float().to(device)
  Y_single_data = mnist_test.test_labels[r:r+1].to(device)

  print(f"Label : {Y_single_data.item()}")
  single_prediction = model(X_single_data)
  print(f"Prediction : {torch.argmax(single_prediction, 1).item()}")

  plt.imshow(mnist_test.test_data[r:r+1].view(28, 28), cmap='Greys', interpolation='nearest')
  plt.show()


