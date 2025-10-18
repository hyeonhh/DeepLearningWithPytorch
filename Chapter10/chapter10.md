# Chapter 10 여러 데이터 소스를 통합 데이터셋으로 합치기
- 해야할 일
  - 원본 데이터를 읽어들이고 전처리하는 루틴을 만들어야한다.
  - <img width="703" height="414" alt="image" src="https://github.com/user-attachments/assets/d20c43b9-bce2-420b-864b-954e28d613b3" />
  - 우리의 목표 : 원본 CT 스캔 데이터와 데이터에 달아놓은 어노테이션 목록으로 훈련 샘플을 만드는 것 
  - <img width="637" height="380" alt="image" src="https://github.com/user-attachments/assets/2deafa1c-a3b9-4739-9987-02fdcc63dc35" />


## 10.1 원본 CT 데이터 파일
- CT 데이터 : 메타데이터 헤더 정보가 포함된 `.mhd` 파일 + 3차원 배열을 만들 원본 데이터 바이트를 포함하는 `.raw파일`
- 각 파일의 이름은 시리즈 UID로 시작한다.
  - 만약 시리즈 UID가 1.2.3인 경우 ->  1.2.3.mhd, 1.2.3.raw
- CT 클래스는 두 파일을 읽어서 3차원 배열을 만들고 환자 좌표계를 배열에서 필요로 하는 인덱스 , 행, 열 좌표로 바꿔주는 변환 행렬도 만든다.
- 어노테이션 파일도 읽어야한다.
  - 각 결절의 좌표 목록, 악성 여부, 해당 CT 스캔의 시리즈 UID 등등
  - 결절 좌표가 좌표계 변환을 거치면 결절의 중심에 해당하는 복셀의 인덱스, 행, 열 정보가 생긴다.
- IRC 좌표를 사용해 CT 데이터의 작은 3차원 부분 단면을 얻어 모델에 대한 입력으로 사용할 수 있다.
## 10.2 LUNA 어노테이션 데이터 파싱
<img width="566" height="423" alt="image" src="https://github.com/user-attachments/assets/8f1ca24c-db31-4fce-8b9e-e52f4c587f82" />
- 좌표 정보, 해당 좌표 지점이 결절인지 여부, 스캔에 대한 고유 식별자를 얻을 것이다.
- 
