MLP

개요
다층 퍼셉트론은 입력층, 은닉층, 출력층으로 구성된 신경망 모델
단층 퍼셉트론의 한계를 극복하기 위해 층을 여러 개 쌓은 구조

MLP 사용 이유
단층 퍼셉트론의 한계 : y=ax+b 형태의 선형 방정식만을 이용하여 문제를 해결하려 함. 직선 하나로 데이터를 나눌수 없는 경우에는 학습이 불가능

Ex). XOR 문제

(1)
<img width="319" height="213" alt="XOR" src="https://github.com/user-attachments/assets/6ee4d3cd-9a6f-44b9-b977-fccfded83faf" />

(2)
<img width="329" height="314" alt="XOR_graph" src="https://github.com/user-attachments/assets/5fa5884c-3b4d-4673-b2b0-2466e86258e9" />

2번 이미지의 그래프를 보면은 검정점과 흰색 점을 단순히 하나의 선형으로 나눌 수가 없는 문제가 생김
이러한 문제를 해결하고자 층을 여러 개 쌓고 활성화 함수를 적용한 다층 퍼셉트론(MLP)를 고안


구조
입력층 : 그냥 데이터 입력하는 곳
은닉층 : 데이터의 특징을 추출하고 학습하는 곳
출력층 : 최종적인 결과를 추출하는 곳


(3)
<img width="174" height="70" alt="image" src="https://github.com/user-attachments/assets/a4d71067-7aef-4c7e-aab5-89e2e5abfc89" />

위의 이미지와 같이 입력층에서 x값을 넣게 되면은 가중치 W를 곱한 후 bias를 더하여 활성화 함수에 넣는 방식으로 각 층에서 계산


순전파(Feedforward)
이렇게 Feedforward를 통해 나온 예측값을 통해서 정답(label)을 손실함수(CrossEntropy)에 넣어 오차를 구함

역전파(Backpropagation)
역전파는 모델이 출력한 예측값과 실제 정답 사이의 오차를 줄이기 위해 W(가중치), bias(편향)을 수정하는 과정
편미분을 이용하여 각 변수의 값이 변화할 때 손실 함수에 어떠 영향을 주는 지 계산 (즉 극솟값을 찾기 위한 기울기 변화)


(4)
<img width="573" height="308" alt="image" src="https://github.com/user-attachments/assets/d0116e47-6e5f-43d6-b3d6-8e8bef5ee457" />

4번 이미지에서 손실함수의 극솟값을 구하려면 기울기를 통하여 가장 낮은 곳인 극솟값을 찾음

기울기를 구하는 이유 : Backpropagation의 목표는 손실함수의 극솟값을 찾는 것. 현재 위치에서 기울기를 구하면 어느 방향으로 가야 손실함수가 작아지는지 알 수 있다. 따라서 W(가중치) 업데이트를 위해 기울기를 사용


