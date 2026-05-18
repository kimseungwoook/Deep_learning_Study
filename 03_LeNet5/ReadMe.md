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
