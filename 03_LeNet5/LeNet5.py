import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pytz import timezone

device = "cuda" if torch.cuda.is_available() else "cpu"

randomseed = 42
learning_rate = 0.001
batch_size = 32
epochs = 15

image_size = 32
n_classes = 10

def get_accuracy(model, data_loader, device):
  correct_pred = 0
  n = 0
  with torch.no_grad():
    model.eval()
    for x, y in data_loader:
      x = x.to(device)
      y = y.to(device)
      _, pred = model(x)
      _, pred_labels = torch.max(pred, 1)

      correct_pred += (pred_labels == y).sum()
      n += y.size(0)
    return correct_pred.float() / n

def plot_loss(train_loss, val_loss):
  plt.style.use('grayscale')
  train_loss = np.array(train_loss)
  val_loss = np.array(val_loss)
  fig, ax = plt.subplots(1, 1, figsize = (8,4.5))
  ax.plot(train_loss, color = 'green', label='Training Loss')
  ax.plot(val_loss, color = 'red', label='Validation Loss')
  ax.set(title = 'Loss Over Epocjs', xlabel='Epoch', ylabel='Loss')
  ax.legend()
  fig.show()
  plt.style.use('default')

def train(train_loader, model, criterion, optimizer, device):
  model.train()
  running_loss = 0
  for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    _, pred = model(x)
    loss = criterion(pred, y)
    running_loss += loss.item() * x.size(0)
    loss.backward()
    optimizer.step()
  epoch_loss = running_loss / len(train_loader.dataset)
  return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):
  model.eval()
  running_loss = 0

  for x, y in valid_loader:
    x = x.to(device)
    y = y.to(device)
    _, pred = model(x)
    loss = criterion(pred, y)
    running_loss += loss.item() * x.size(0)

  epoch_loss = running_loss / len(valid_loader.dataset)
  return model, epoch_loss

def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
  best_loss = 1e10
  train_loss = []
  val_loss = []

  for epoch in range(0, epochs):
    model, optimizer, t_loss = train(train_loader, model, criterion, optimizer, device)
    train_loss.append(t_loss)

    # valide
    with torch.no_grad():
      model, v_loss = validate(valid_loader, model, criterion, device)
      val_loss.append(v_loss)

    if epoch % print_every == (print_every - 1):
      train_acc = get_accuracy(model, train_loader, device)
      valid_acc = get_accuracy(model, valid_loader, device)

      print(datetime.now(timezone('Asia/Seoul')).time().replace(microsecond=0), '--- ',
        f'Epoch: {epoch}\t',
        f'Train loss: {t_loss:.4f}\t',
        f'Valid loss: {v_loss:.4f}\t',
        f'Train accuracy: {100 * train_acc:.2f}\t',
        f'Valid accuracy: {100 * valid_acc:.2f}')

  plot_loss(train_loss, val_loss)
  return model, optimizer, (train_loss, val_loss)

class LeNet5(nn.Module):
  def __init__(self, n_classes):
    super(LeNet5, self).__init__()

    self.feature_extractor = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
        nn.Tanh()
    )

    self.classifier = nn.Sequential(
        nn.Linear(in_features=120, out_features=84),
        nn.Tanh(),
        nn.Linear(in_features=84, out_features=n_classes),
    )

  def forward(self, x):
    x = self.feature_extractor(x)
    x = torch.flatten(x, 1)
    logits = self.classifier(x)
    probs = F.softmax(logits, dim=1)
    return logits, probs

if __name__ == "__main__":
  transforms = transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor()
  ])

  train_dataset = datasets.MNIST(root='MNIST_dataset', train=True, transform=transforms, download=True)
  valid_dataset = datasets.MNIST(root="MNIST_dataset", train=False, transform=transforms, download=True)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

  torch.manual_seed(randomseed)

  model = LeNet5(n_classes=n_classes)
  model = model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss()

  model, optimizer, losses = training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device)
