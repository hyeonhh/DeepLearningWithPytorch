# Chapter 11 : 종양 탐지를 위한 분류 모델 훈련
## 11.1 기본 모델과 훈련 루프
## 11.2 애플리케이션의 메인 진입점
## 11.3 사전 훈련 설정과 초기화
- 에포크 내의 각 배치를 순환하기에 앞서 초기화가 필요하다.
- <img width="740" height="537" alt="image" src="https://github.com/user-attachments/assets/f3969992-4c31-45a2-9508-20c736dcf4ba" />
  - 2가지 초기화 작업이 필요하다.
      1. 모델과 옵티마이저의 초기화
      2. Dataset과 DataLoader 인스턴스 초기화
  - LunaDataset은 랜덤으로 선택된 샘플셋을 정의하여 훈련 에포크를 채워줄 것
  - DataLoader 인스턴스는 데이터셋으로부터 데이터를 읽는 작업을 수행하여 애플리케이션에 제공한다.
### 11.3.1 모델과 옵티마이저 초기화
```python
import argparse
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class TrainingApp:
    def __init__(self, sys_argv = None):
        if sys_argv is None:
            sys_argn = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help = 'Number of worker processes for background data loading',
                            default = 8,
                            type = int,
                           )
        parser.add_argument('--batch_size',
                           help = 'Batch size to use for training',
                            default = 32,
                            type = int
                           )
        parser.add_arguemnt('--epochs',
                            help = 'number of epochs to train for',
                            default =1,
                            type = int
                           )
        parser.add_argument('--tb-prefix',default='p2ch11',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='dwlpt',
        )
        self.cli_args = parser.parser_args(sys.argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0
        self.use_cuda = torch.cuda.is_available() 
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        
    def initModel(self):
        model = RSNAModel() #만들어야함.
        if self.use_cuda:
            log.info("Using CUDA; {} devices".format(torch.cuda.device_count()))

            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr = 0.001, momentum = 0.99) # 잘 안된다면 학습률을 0.01이나 0.0001로 바꿔 시도해보자 
        # momentum : 모멘텀은 SGD에서 계산된 접선의 기울기에 한 시점(step) 전의 접선의 기울기값을 일정한 비율만큼 반영한다.
        
  ```
  
### 11.3.2 데이터 로더의 관리와 데이터 공급
- 앞서 10장에서 만든 LunaDataset 클래스는 우리의 원본 데이터와 파이토치 빌딩 블럭을 위해 구조화된 텐서 사이를 이어주는 다리 역할을 해준다.
  - torch.nn.Conv3d와 CT의 3차원 데이터는 완전히 다르다.
    - torch.nn.Conv3d 는 5차원 입력(N,C,D,H,W)인 샘플 수와 샘플 당 채널 수, 깊이, 높이, 너비까지 입력해야한다.
- <img width="738" height="535" alt="image" src="https://github.com/user-attachments/assets/8981f205-a71d-46de-b723-6a3f488d2e79" />
- 단일 샘플에 대한 작업 수행은 비효율적이다. 여러 플랫폼에서 이미 여러 샘플에 대한 병렬 계산을 지원하기 때문
- 따라서 여러 샘플을 배치 튜플로 묶어 한번에 처리할 수 있게 해야한다.

```python
def initTrainDl(self):
  train_ds = RSNADataset(
    val_stride=10,
            isValSet_bool = False,
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        train_dl = DataLoader(
            train_ds,
            batch_size = batch_size, #알아서 배치로 나뉜다. 
            num_workers = self.cli_args.num_workers,
            pin_memory - self.use_cuda, # 고정된 메모리 영역이 GPU 쪽으로 빠르게 전송된다.
        )
  return train_dl
        
```
- 데이터 로더는 개별 샘플을 배치로 만들 뿐만 아니라,별도의 프로세스와 공유 메모리를 사용한 병렬 로딩도 제공한다.
- 데이터 로더 객체를 만들 때 num_workers=...만 지정해주면 위의 그림처럼 배치를 만들어준다.

## 11.4 첫 번째 신경망 설계 ~!
- <img width="723" height="552" alt="image" src="https://github.com/user-attachments/assets/ca95f908-18d9-49c1-91f5-61be34132625" />
### 11.4.1 핵심 컨볼루션
- 분류모델에서는 테일, 백본(바디), 헤드로 구성된 구조가 흔하다.
- 테일
  - 입력을 신경망에 넣기 전 처리 과정을 담당
  - 백본이 원하는 형태로 입력을 만들어야한다. 
- 백본
  - 일반적으로 연속된 블럭이 배치된다.
  - 각 블럭은 동일한/유사한 세트의 계층을 가지며 블럭과 블럭을 거칠 때마다 필요한 입력 크기나 필터 수가 달라진다.
  - 여기서는 3*3 conv 1개, 하나의 활성화(ReLU), 블록 끝에 max pooling 연산이 이어진 블럭을 사용한다.
- 헤드
  - 백본의 출력을 받아 원하는 출력 형태로 바꾼다.
    - 중간 출력물을 평탄화(Flattening)하기도 하고 완전 연결 계층(fully connected layer)에 전달하는 역할을 하기도 한다. 
    - <img width="622" height="312" alt="image" src="https://github.com/user-attachments/assets/56f91fe5-b8a9-47a9-9c01-06a291ab250e" />
    
```
It gets interesting when we use two 3 × 3 × 3 convolutions stacked back to back. Stack-
ing convolutional layers allows the final output voxel (or pixel) to be influenced by an
input further away than the size of the convolutional kernel suggests
```

- 2개의 3*3*3 컨볼루션이 연속으로 쌓이면
  - 첫 번째 컨볼루션의 출력 복셀 하나도 결국 입력 3*3*3에 대응하고,
  - 두 번째 컨볼루션도 그 전 단계의 3*3*3 출력을 한번 더 합성곱한다.
  - 최종 출력의 한 점은 바로 전 레이어의 3*3*3 점에 의존, 실제로는 입력의 5*5*5 영역에 영향을 준다.
  - 그 각각도 다시 입력의 3*3*3 영역에서 나왔으므로 최종 출력 한 점이 처음 입력에서 참조하는 영역(receptive field)은 더 넓어진다.
  - <img width="1119" height="319" alt="image" src="https://github.com/user-attachments/assets/2c4e57c5-f1da-4caa-87f5-5c3022ec5afd" />

- 각 3*3*3 컨볼루션은 수용 필드에 대해 모서리당 한 개의 복셀 경계를 더한다.
  - 2*2 출력은 4*4 수용 필드를 가지고, 다시 6*6 수용 필드를 가진다.
    - 결과적으로 두 개로 쌓인 3*3*3 계층은 5*5*5 컨볼루션보다 더 적은 파라미터를 가진다.
  - receptive field = 1 + (커널 크기 -1) * 레이어 수
    - 두 번 쌓으면 1 + (3-1) * 2 = 5
  - 파라미터 수 비교
    - 5*5*5 커널을 한번 쓰면 파라미터 수는 5*5*5 = 125개
    - 3*3*3 커널을 2개 쓰면 파라미터 수는 3*3*3 * 2 = 54개이다.
    - 즉 겹겹이 쌓인 작은 커널이 더 넓은 수용 필드를 만들면서도 파라미터는 훨씬 적게 사용한다.
- 두 개로 쌓인 컨볼루션의 출력은 2*2*2 맥스 풀링으로 들어간다.
  - 6*6*6 수용 필드로부터 가장 큰 값을 가지는 한개의 5*5*5 필드를 만드는 것이다.
- 이러한 블럭을 여러 번 반복해서 모델의 백본을 구성한다.
### 11.4.2 전체 모델
