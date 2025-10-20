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

### 10.2.2 어노테이션 데이터와 후보 데이터 합치기
- getCandidateInfoList()
  - `@functools.lru_cache(1) `
    - 인자값이 해시 가능한 타입인지 확인한다.
      - 인자값 : 함수 호출 시 함수에 전달되는 매개변수
      - 해시 가능 : 그 값이 프로그램 실행동안 변하지 않고 일관된 해시값을 가질 수 있는 객체를 의미한다. 
    - 이전에 같은 인자조합으로 호출한 적이 있다면, 저장된 결과를 바로 반환한다.
    - 캐시에 없는 인자 조합이면 함수를 실행한 후 결과를 캐시에 저장한다.
    - 캐시 크기(max size)를 넘을 경우, 가장 오랫동안 사용되지 않은 항목이 제거된다.
  - 우리는 `requireOnDisk_bool  = True ` 인자를 사용함
  - `glob.glob `조건에 맞는 데이터를 리스트 형태로 제공한다.
  - `dict.setdefault`
    - https://wikidocs.net/33177
  - 여기서는 `annotations.csv`와 `candidates.csv` 데이터를 통해,
    - `annotations.csv`에서는 `series_uid`, `annotationCenter_xyz`, `annotaionDiameter_mm` 정보를 얻는다.
    - `candidates.csv` 에서는 `candidateCenter_xyz` 정보를 얻는다.
    - 그리고 `annotationCenter_xyz와` `candidateCenter_xyz` 사이의 거리를 비교하여
      - 가까우면 `annotaionDiameter_mm를` `candidateDiameter_mm` 정보에 할당해준다.
      - 아니라면 `candidateDiameter_mm에` 0.0을 할당해준다.
## 10.3 개별 CT 스캔 로딩
- 디스크에서 CT 데이터를 얻어와 파이썬 객체로 변화해서 3차원 결절 밀도 데이터로 사용할 수 있도록 만드는 작업이다.
- <img width="540" height="424" alt="image" src="https://github.com/user-attachments/assets/4c012361-db2a-4e91-8ba1-88a36ccab0a6" />
- 대용량 원본 데이터 중 필요한 데이터만 찾아볼 수 있도록 걸러낼 방법을 생각해보자.
- 파서 찾아보자.
- LUNA 데이터셋은 `MetaIO` 포맷을 사용함
  - MHA/MHD (.mha, .mhd 포맷)
    - 다차원 이미지를 지원하며 해당 이미지와 관련된 메타 데이터를 태그-값 텍스트 형식으로 설명한다.
    - .mhd : 텍스트 기반 헤더 파일
    - .mha : 데이터와 헤더를 모두 포함하는 바이너리 파일
    - `class Ct의 __init__ 메서드`에서 `SimpleITK` 을 통해 이미지를 가져온다.
      - `ct_mhd = SimpleITK.getImage(이미지 경로)`
      - `np.array(SimpleITK.GetArrayFromImage(ct_mhd)`
- HU 단위
  -  `ct_a.clip(-1000, 1000, ct_a)`
    - -1000: 공기
    - 1000 : 뼈나 금속에 해당하는 밀도값
    - 종양은 대부분 0HU 근처이다.
- 따라서 데이터의 범위는 -1000 ~ 1000 사이이다.
## 10.4 환자 좌표계를 사용해 결절 위치 정하기
- 딥러닝 모델은 고정된 크기의 입력이 필요하다.
  - 입력 뉴런 수가 고정되어있기 때문
- 분류기의 입력으로 사용할 고정된 크기의 결절 후보를 담을 배열을 만들 수 있어야한다.
- 입력의 어느 위치에 후보가 있는지 따로 학습할 필요 없도록 항상 중심에 후보를 위하도록 crop한다.
### 10.4.1 환자 좌표계 : The patient coordinate system
- 앞서 candidate center data는 복셀이 아니라 밀리미터 단위로 표시되어있다.
  - 그러나 mm로 나타낸 위치 정보를 배열 인덱스로 사용하면 원하는대로 작업할 수 없다.
    - 왜냐면??!!
      - 의료 영상 데이터는 3D 배열로 저장되며 각 패열의 인덱스는 복셀 단위이다.
      - 각 복셀은 실제 공간에서 일정한 크기를 가진다.
      - mm단위는 실제 환자의 몸의 위치를 의미하지만 배열 인덱스는 몇 번째 복셀인가를 의미한다.
      - 따라서 mm 좌표를 배열 인덱스로 사용하려면 (mm - 원점 )/복셀 크기 변환이 필요하다.
      - ex :
        - 예를 들어, 복셀 크기가 (1.0, 1.0, 1.0)mm이고, 원점이 (0, 0, 0)mm라면 (10.0, 20.0, 30.0)mm 위치는 배열 인덱스 (10, 20, 30)에 해당한다. 
        - 하지만 복셀 크기가 (0.7, 0.7, 1.0)mm라면 (10.5, 14.0, 5.0)mm 위치는 배열 인덱스 (15, 20, 5)가 된다.
        
- <img width="522" height="420" alt="image" src="https://github.com/user-attachments/assets/988c2b1f-3d0c-4402-9cb5-459c8d5eb136" />

- mm 기반 좌표계인 (X,Y,Z)로부터 CT 스캔 단면 데이터 배열에서 사용한 복셀 주소 기반 좌표계인 (I, R , C)로 좌표를 변환해야한다.

- <img width="539" height="307" alt="image" src="https://github.com/user-attachments/assets/707c814d-d046-46f4-a5d0-7d038b7d5032" />
- X값은 right에서 left로 갈수록 증가
- Y값은 anterior 에서 posterior로 갈수록 증가
- Z값은 inferior에서 superior로 갈수록 증가
  
<img width="519" height="280" alt="image" src="https://github.com/user-attachments/assets/3d9dd7c9-dfd3-43e3-b986-1ad8d7b7632e" />
- 환자 좌표계는 mm 단위로 측정되며 위치 기준을 임의로 잡기 때문에 위 그림처럼 CT 복셀 배열 기준과 일치하지 않는다. 

- 환자 좌표계는 특정 스캔과 무관하게 관심 있는 위치를 지정하기 위해 사용되어왔다.
- CT 배열과 환자 좌표계 사이의 관계를 정의하는 메타 데이터는 DICOM의 헤더에 저장되어있다.
  - 이 메타데이터를 통해 XYZ -> IRC 변환이 가능하다.
  -  메타 데이터
    - ImagePosition(Patient) : 한 슬라이스(이미지)의 왼쪽 위 모서리(첫번째 픽셀)의 환자좌표계 (XYZ, mm)위치를 나타낸다.
      - 이 값이 CT 배열의 원점 역할을 한다.
    - ImageOrientation(Patient) : 이미지의 행과 열이 환자 좌표계의 어떤 방향을 가리키는지를 나타내는 6개의 값(3D 단위 벡터 2개)
    - Pixel Spacing : 한 픽셀(복셀)의 실제 크기 (mm단위)를 나타낸다.
    - Slice Tinkness / Spacing Between Slices : 슬라이스 간 간격(복셀의 z축 크긱)
      - 3D 볼륨에서는 이 값이 z축 방향 spacing으로 사용된다.
### 10.4.2 Ct scan shape와 voxel sizes
- 일반적으로 복셀은 정육면체가 아니다. 1.125mm * 1.125mm * 2.5mm 크기 혹은 이와 유사한 크기를 가진다.
- <img width="675" height="252" alt="image" src="https://github.com/user-attachments/assets/52411a18-e860-418e-bf99-6d5a0853d759" />

- CT는 일반적으로 512 * 512 로 구성되며 인덱스 차원은 대략 총 100 ~ 250개의 단면으로 이루어진다.
- 각 CT는 파일 메타 데이터 내에 복셀의 크기를 mm 단위로 정의하고, 이를 참조하기 위해 ct_mhd.GetSpacing()을 호출한다.

### 10.4.3 mm를 voxel 주소로 변환하기
#### 복셀 인덱스를 좌표로 바꾸기
1. 좌표를 XYZ 체계로 만들기 위해 IRC에서 CRI로 뒤집는다.
2. 인덱스를 복셀 크기로 확대 축소한다.
3. 파이썬의 @를 사용하여 방향을 나타내는 행렬과 행렬곱을 수행한다.
4. 기준으로부터 오프셋을 더한다.
* xyz에서 irc로 바꾸려면 각 단계를 역순으로 실행하되 순서도 거꾸로 진행하면 된다.

##### 만약 z값 데이터를 모른다면 아래 코드를 참조해서 z-position을 구하자
- <img width="539" height="505" alt="image" src="https://github.com/user-attachments/assets/311b50d4-a1e9-4135-93eb-38c826c129f7" />

##### 다시 10.4.3 내용으로 넘어가서
- <img width="602" height="352" alt="image" src="https://github.com/user-attachments/assets/8c022784-02f3-4d89-981a-11168945f364" />

**IRC -> XYZ**
- `cri_a = np.array(coord_irc)[::-1]`  : 1. 넘파이 배열로 변환하며 순서를 바꿔준다.
- 'origin_a' : DICOM 헤더에서 추출한 원점 좌표(XYZ, mm)
- 'vxSize_a' : 각 축의 복셀 크기(mm)
  - `[col_spacing, row_spacing, slice_spacing] `
- `direction_a` : 이미지 방향 행렬(보통 3*3 행렬)
  - 이미지 배열과 환자 좌표계 축이 정확히 어떻게 정렬되는지 나타낸다.
- 변환 과정
  1. cri_a * vxSize_a : IRC 좌표 별로 복셀 사이즈를 곱해서 실제 mm 거리로 변환한다.
  2. 환자 좌표계와의 정렬 적용
    - direction_a @ : 방향 행렬을 곱해 배열 mm 좌표를 환자 좌표계로 매핑한다.
      - 이미지 행/열 슬라이드(IRC) 축이 실제 XYZ 공간의 어느 방향에 대응되는지를 정의하고, IRC 좌표의 mm 위치를 환자 좌표계 XYZ로 회전 또는 정렬해주는 작업을 수행한다.
  3. 절대 위치 보정
    - 결과에 다 원점(origin_a)를 더해서 데이터 전체의 기준 좌표에 상대적이었던 값을 화낮 기준 절대 좌표(XYZ,mm)로 만든다.
- 이 공식은 CT/MRI 배열에서 특정 위치가 실제 환자 몸의 어디에 대응되는지 계산할 때 필요하다.


**XYZ -> IRC**
1. (xyz좌표 - 원점좌표)를 통해 원점으로부터의 상대 위치를 구해준다.
2. 앞의 결과에 방향행렬의 역행렬을 행렬 곱을 해준다.
3. 앞의 결과에서 vxSize_a를 나눠준다.
4. 정수로 변환하기 전에 적절히 반올림을 해준다.(np.round)

- 왜 방향행렬의 역행렬을 곱해야할까?
  - IRC에서 XYZ로 갈 때는 방향 행렬을 곱해서 "배열 공간을 환자 공간으로 회전"한다.
  - 반대로, XYZ에서 IRC로 갈 때는 이 과정을 거꾸로(역방향) 적용해야 하므로, 방향 행렬의 역행렬을 곱해야 한다.

### 10.4.4 CT 스캔에서 결절 추출하기 
- <img width="644" height="444" alt="image" src="https://github.com/user-attachments/assets/c3dd2e53-67fa-407c-a0b1-79c0368fdafd" />
- `getRawCandidate` : LUNA CSV 데이터에 명시된 환자 좌표계(X,Y,Z)로 표시된 중심 정보와 복셀 단위의 너비 정보(width_irc)도 인자로 전달받아 정육면체의 CT 덩어리와 배열 좌표로 변환된 후보의 중심값을 반환한다.
- center_irc 데이터 출력 형태 확인하기
## 10.5 간단한 데이터셋 구현
- `Ct` Class : 다양한 Ct 샘플들
- `LunaDataset Dataset` : Ct 샘플들을 정규화하고 각 Ct의 결절은 Flatten작업을 통해 어느 Ct 객체에서 가져온 샘플인지에 상관없이 인출가능하도록 하나의 단일 컬렉션으로 합쳐진다.
  - 반환값 살펴보기
    - <img width="339" height="118" alt="image" src="https://github.com/user-attachments/assets/8889f289-1825-4b22-9282-f49c062d3049" />
    - <img width="748" height="211" alt="image" src="https://github.com/user-attachments/assets/a10f4fbd-8d86-42da-898b-aaaddefce5e8" />


### 10.5.1 getCtRawCandidate 함수로 후보 배열 캐싱하기
#### On-disk caching
  -   <img width="678" height="170" alt="image" src="https://github.com/user-attachments/assets/ccdc13af-7879-4c4e-8ca7-3bcc90fa30ac" />
  - 모든 샘플에 대해 디스크로부터 전체 Ct 스캔을 읽는 부담을 피할 수 있다.
  - 메모리 캐싱
    - `@functools.lru_cache(1, typed = True)` 
    - getCt의 반환값을 메모리에 캐싱해서 동일한 Ct 인스턴스에 대한 요청은 디스크에서 모든 데이터를 다시 읽을 필요가 없다.
    - 같은 series_uid로 여러번 요청하면 이미 메모리에 있는 Ct 객체를 바로 반환하므로 디스크에서 다시 읽지 않아도 되어 매우 빠르다. 
    - 메모리에 딱 하나의 CT만 있으므로 접근하는 순서를 신경쓰지 않으면 캐시 미스 발생 가능하다.
  - 디스크 캐싱
    - `@raw_cache_memoize(typed = True)`
    - `getCt를` 호출하는 `getCtRawCandidate` 함수도 출력을 캐싱한다.
    -  함수의 결과를 디스크에 저장한다.
      - 한번 계산된 결과는 디스크에서 바로 읽어올 수 있어, getCt는 아예 호출되지 않는다.

### 10.5.2 LunaDatset.__init__으로 데이터셋 만들기
- 훈련셋과 검증셋으로 샘플을 나누기
- <img width="704" height="275" alt="image" src="https://github.com/user-attachments/assets/cd57bf91-ab3b-4c5c-b605-605a329fb910" />
- val_stride 파라미터를 통해 샘플 중 10번째에 해당하는 모든 경우를 검증셋으로 둔다.
- isValSet_bool 파라미터로 훈련 데이터나 검증 데이터만 사용할지, 둘다 사용할지를 결정한다.

### 10.5.3 훈련/검증 분리

### 분류 모델
- 이후 모델에 전달될 데이터를 `LunaDataset을` 사용한다.




