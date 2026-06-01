import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import optim
import torch.nn as nn
import random
import os
import cv2
from PIL import Image
from tqdm.notebook import tqdm
import time
from torchsummary import summary
from matplotlib import pyplot as plt
import pandas as pd
import torch.nn.functional as F
import torchvision.models as models

# 🟢 하드웨어 디바이스 설정 (GPU 사용 가능 여부 체크)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ==========================================
# 🖼️ 1. 데이터 전처리 및 증강(Augmentation) 정의
# ==========================================
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.ImageTransform = {
            "train" : transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.6, 1.0)), # 무작위 크롭 및 확대
                transforms.RandomHorizontalFlip(p=0.5),                # 좌우 반전
                transforms.RandomRotation(degrees=15),                 # 무작위 회전
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 색상 변형
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            "val" : transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase):
        return self.ImageTransform[phase](img)


# ==========================================
# 📂 2. 커스텀 데이터셋(Dataset) 클래스 정의
# ==========================================
class Dog_Cat_dataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        
        # 🟢 에러 방지: 흑백 이미지가 들어와도 강제로 RGB 3채널 컬러로 변환하여 로드
        img = Image.open(img_path).convert("RGB")
        img_transform = self.transform(img, self.phase)

        # 파일명에서 라벨 추출 (소문자 변환 후 비교)
        filename = img_path.split('/')[-1].lower()

        if "dog" in filename:
            label = 0
        elif "cat" in filename:
            label = 1
        else:
            label = 0

        return img_transform, label


# ==========================================
# 👟 3. 모델 학습 및 검증 함수 정의 (시각화 기능 포함)
# ==========================================
def train_model(model, dataloader_dict, criterion, optimizer, num_epochs):
    since = time.time()
    
    # 🟢 에포크별 Loss, Acc 기록용 저장소 생성 (시각화 목적)
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()

            # 🟢 매 에포크, 매 페이즈 시작 시 누적 변수 초기화 위치 교정
            epoch_loss = 0.0
            epoch_corrects = 0

            # 배치 단위 연산 수행
            for inputs, labels in tqdm(dataloader_dict[phase], desc=f"{phase} phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 배치별 손실값 및 정답 개수 누적
                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)

            # 🟢 에포크 종료 후 평균값 계산 및 출력 위치 교정 (배치 루프 바깥)
            dataset_size = len(dataloader_dict[phase].dataset)
            epoch_loss_avg = epoch_loss / dataset_size
            epoch_acc_avg = epoch_corrects.double() / dataset_size

            print("{} Loss : {:.4f}   Acc : {:.4f}".format(phase, epoch_loss_avg, epoch_acc_avg))
            
            # history 딕셔너리에 데이터 기록
            if phase == 'train':
                history['train_loss'].append(epoch_loss_avg)
                history['train_acc'].append(epoch_acc_avg.item())
            else:
                history['val_loss'].append(epoch_loss_avg)
                history['val_acc'].append(epoch_acc_avg.item())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, history


# ==========================================
# 🚀 4. 메인 실행 제어부
# ==========================================
if __name__ == "__main__":
    # 기본 하이퍼파라미터 및 경로 설정
    size = 227
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32

    dog_directory = "./drive/MyDrive/data/dog_cat_dataset/dogs"
    cat_directory = "./drive/MyDrive/data/dog_cat_dataset/cats"

    dog_image_filepath = sorted([os.path.join(dog_directory, i) for i in os.listdir(dog_directory)])
    cat_image_filepath = sorted([os.path.join(cat_directory, i) for i in os.listdir(cat_directory)])

    Image_filepath = [*dog_image_filepath, *cat_image_filepath]
    correct_Image_filepath = [i for i in Image_filepath if cv2.imread(i) is not None]

    # 데이터셋 셔플 및 분할
    random.seed(42)
    random.shuffle(correct_Image_filepath)

    train_image_filepath = correct_Image_filepath[:420]
    val_image_filepath = correct_Image_filepath[420:-10]
    test_image_filepath = correct_Image_filepath[-10:]

    print(f"train_image 개수 : {len(train_image_filepath)}")
    print(f"val_image 개수 : {len(val_image_filepath)}")
    print(f"test_image 개수 : {len(test_image_filepath)}")

    # Dataset 객체 인스턴스 생성
    train_dataset = Dog_Cat_dataset(train_image_filepath,
                                    transform=ImageTransform(size, mean, std),
                                    phase='train')
    val_dataset = Dog_Cat_dataset(val_image_filepath,
                                  transform=ImageTransform(size, mean, std),
                                  phase="val")
    test_dataset = Dog_Cat_dataset(test_image_filepath,
                                   transform=ImageTransform(size, mean, std),
                                   phase='val')

    # DataLoader 정의
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataloader_dict = {'train' : train_dataloader, 'val' : val_dataloader}

    # 🟢 1. 사전 학습된 오리지널 ImageNet AlexNet 가중치 로드
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

    # 🟢 2. 최종 출력 레이어 차원을 2(개, 고양이)로 수정 및 디바이스 할당
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 2)
    model.to(device)

    # 🟢 3. 옵티마이저 및 손실함수 세팅 (전이 학습 미세조정용 Adam 알고리즘 및 적정 LR 사용)
    optim = torch.optim.Adam(model.parameters(), lr=0.00002)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    summary(model, input_size=(3, 227, 227))

    # 모델 학습 진행 및 기록 데이터 반환
    model, history = train_model(model, dataloader_dict, criterion, optim, num_epochs)

    # ==========================================
    # 📊 5. 보고서용 훈련 결과 추이 그래프 시각화
    # ==========================================
    plt.figure(figsize=(14, 5))
    epochs_range = range(1, num_epochs + 1)

    # 손실값(Loss) 추이 시각화
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o', color='crimson')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='x', linestyle='--', color='darkblue')
    plt.title('AlexNet - Training & Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    # 정확도(Accuracy) 추이 시각화
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o', color='forestgreen')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc', marker='x', linestyle='--', color='darkorange')
    plt.title('AlexNet - Training & Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 📝 6. 실전 테스트 데이터 검증 및 CSV 출력
    # ==========================================
    id_list = []
    dog_pred_list = []
    cat_pred_list = []

    with torch.no_grad():
        for test_path in tqdm(test_image_filepath, desc="Testing phase"):
            img = Image.open(test_path).convert("RGB")
            label = test_path.split('/')[-1].split('.')[0].split('_')[0]
            
            transform = ImageTransform(size, mean, std)
            img = transform(img, phase='val')
            img = img.unsqueeze(0).to(device)

            model.eval()
            outputs = model(img)
            preds = F.softmax(outputs, dim=1)

            id_list.append(label)

            # 🟢 4. [0, 0] 인덱싱과 .item() 조합으로 안전하게 리스트 내 중복 제거 후 단일 실수값 추출
            dog = preds[0, 0].item()
            cat = preds[0, 1].item()

            dog_pred_list.append(dog)
            cat_pred_list.append(cat)

        # 데이터프레임 빌드 및 저장
        res = pd.DataFrame({
            'id' : id_list,
            'Dog_pred' : dog_pred_list,
            'Cat_pred' : cat_pred_list
        })

        res.to_csv('./drive/MyDrive/data/dog_cat_dataset/AlexNet.csv', index=False)
        print("\n--- 최종 예측 결과 샘플 ---")
        print(res.head(5))
