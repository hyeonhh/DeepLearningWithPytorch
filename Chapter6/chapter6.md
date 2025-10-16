# Chapter 6 신경망을 활용한 데이터 적합
## 6.1 인공 뉴런
- 뉴런은 단순히 입력에 대한 선형 변환과 activation function(활성함수)라는 비선형 함수를 적용하는 역할을 한다.
- 일반적으로 입력 x나 출력 o는 단순한 스칼라값 or 여러 스칼라값을 가진 벡터
- w는 단일 스칼라 혹은 행렬
- b는 스칼라 혹은 벡터일 수 있다.
- 행렬과 벡터 차이
  - 벡터 : 1차원 배열 `w = [w1, w2, w3]`
  - 행렬 : 2차원 이상 
    w =
       ` [ w11, w12, w13]
        [ w21 ,w22, w23]`
    

### 6.1.2 오차 함수 
- 선형 모델과 딥러닝의 차이는 the shape of the error function(오차 함수의 모양)
- 선형 모델 : 파라미터의 조정이 하나의 정답에 가까워지도록 값을 추정한다 -> updates parameters attempting to estimate singular correct answer
- 신경망 : 우리가 근사하려는 각 파라미터에 정답 없다. 대신 모든 뉴런이 협력해서 유용한 출력을 만들기 위한 파라미터를 획득하도록 한다.

- 신경망의 오차 함수는 볼록 형태가 아니다. 이는 activation function(활성함수) 때문!.

### 6.1.3 우리에게 필요한 건 활성 함수



### 6.1.6 신경망에서의 학습의 의미 
- 훈련되지 않은 일반 모델에서 출발하며 우리는 여러 입출력쌍 예제와 역전파할 손실 함수를 제공하여 일반 모델을 거쳐 특정 작업에 최적화한다.
- specializing a generic model to a task using examples 
- 예제를 통해 일반 모델을 최적화하는 과정을 learning이라고 한다.

## 6.3 드디어 신경망
### 6.3.1 선형 모델 대체하기
- 코드 참조
- model.backward()를 호출하면 grad에서 파라미터(미분값)이 추출되고 optimizer는 optimizer.step() 호출 과정에서 적절하게 파라미터 값을 조정한다.
