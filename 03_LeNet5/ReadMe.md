LeNet5는 얀 르쿤 교수가 Backpropagation을 결합하여 자동으로 가중치를 학습하는 현대적인 완성형 CNN모델이다.
이전에 CNN의 특징에 대해서 설명을 하였지만 추가적으로 CNN의 특징에 대해서 설명하겠다.

- Receptive field
  <img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/802b78d3-77ef-4700-9e15-63a9df704ea6" />

  Receptive field는 Output의 pixel 하나가 Input의 어느정도 범위의 정보를 담고 있는가를 의미한다.
  CNN은 Receptive field를 local로 제한하고 local_feature를 추출한다.
  즉, 이미지 전체를 한 번에 처리하지 않고 Receptive field크기 만큼 국소적으로 연결을 제한하기 때문에 파라미터 수를 획기적으로 줄일 수 있으며,
  레이어의 깊이가 깊어질 수록 더 고차원적인 정보를 학습할 수 있게 된다.


- Weight Share
  
  CNN에서는 입력 이미지를 filter를 사용하여 이미지 전체를 훑어 특징을 뽑아내어 output_feature_map을 만든다.
  이때 같은 filter를 사용하여 입력 이미지를 훑는 것을 보아 필터라는 가중치를 공유하는 것을 알 수 있다.
  또한 학습은 Backpropagation에서 filter, bias만을 변경하여 최적의 가중치 값을 찾는다.
  Weight share 덕분에 입력 이미지의 크기가 커져도 parameter 수는 늘어나지 않는다.
  CNN에서 parameter 수가 늘어나는 것은 이미지 크기 때문이 아니라, filter의 크기나 개수를 늘릴 때이다.


- Sub-sampling (pooling)

  Sub-sampling은 오늘날 풀링이라고 불리기도 한다.
  이전에 나는 pooling을 하는 이유는 feature_map의 크기를 줄이지 않으면 나중에 fc_layer에서 파라미터 수가 기하급수적으로 많아지기 때문이라고 설명했었다.
  이는 틀린 말이 아니지만 다른 이유도 존재한다.
  위치가 다른 고양이 이미지 2개가 있다고 가정하자. 기존의 fc_layer 방식은 같은 고양이라도 위치가 다르면 완전히 다른 이미지라고 판단하는 "위치 민감성" 문제가 존재한다.
  pooling은 특정 영역의 대표적인 특징만을 남김으로써 해당 영역 내에 귀가 조금은 움직이더라도 동일한 특징으로 인식을 한다.
  이를 통해 국소적으로 이동을 하는 것은 같은 특징으로 인식하는 불변성을 확보하여 위치 정보에 대한 민감성을 줄여준다.



  - LeNet5
    <img width="1000" height="273" alt="image" src="https://github.com/user-attachments/assets/f6f92576-d17a-44bc-94dd-b248a0f759e1" />

    위의 이미지는 LeNet5의 구조이다.
    input_size = 32
    conv_layer1 : filter_size = 5*5, filter 개수 = 6개, stride = 1 ----> output = 28*28*6
    sub-sampling_layer1 : filter_size = 2*2, filter 개수 = 6개, stride = 2 ----> output = 14*14*6
    conv_layer2 : filter_size = 5*5, filter 개수 = 16개, stride = 1 ----> output = 10x10x16
    <img width="794" height="382" alt="image" src="https://github.com/user-attachments/assets/bbb0db01-2ea9-41f5-8e7e-e50d7cf191ae" />

    여기 conv_layer2에서는 입력 채널 6개를 필터 16개와 모두 연결하지 않았다.
    즉 파라미터 수를 줄여 연산량을 줄여 계산 속도를 빠르게 하기 위해서 그렇다. 또한 모든 채널을 연결하면 각각의 필터 하나가 서로 비슷한 특징을 뽑아내는 16개의 필터가 될 가능성이 높다 이를 해결하기 위해 강제로 각 필터를 서로 다른 채널들과 조합을 이루어 다른 특징들을 뽑아내게 만들어 다양성을 얻게 만든 것이다.

    sub-sampling_layer2 : filter_size = 2*2, filter 개수 = 16개, stride = 2 ----> output = 5*5*16
    conv_layer3 : filter_size = 5*5, filter 개수 = 120개, stride = 1 ----> output = 1*1*120
    fc_layer : input = 1*1*120, output = 1*1*84
    fc_layer : (MNist 데이터 셋은 10개의 숫자를 추측을 하므로 최종 output은 10개로 추론) ----> output = 1*1*10


<img width="833" height="726" alt="image" src="https://github.com/user-attachments/assets/e5adce6a-ee64-4878-a324-05612ba5f0fc" />

  위의 이미지는 코드로 구현을 해보았다.
  오차 : 1.47
  최종 정확도 : 98.65%
  
