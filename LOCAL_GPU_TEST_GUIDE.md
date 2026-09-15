# OMEN RTX 3070 로컬 테스트 안내

Windows PowerShell에서 이 저장소의 최종 CLIP 파이프라인을 실행하는 전체 절차입니다. RTX 3070의 VRAM이 8GB인 경우를 기준으로 `ViT-B/16`을 사용합니다.

## 1. 사전 확인

NVIDIA 드라이버가 설치되어 있는지 확인합니다.

```powershell
nvidia-smi
```

GPU 정보가 출력되지 않으면 OMEN/NVIDIA 드라이버를 먼저 설치하거나 업데이트합니다. CUDA Toolkit을 별도로 설치하지 않아도 PyTorch CUDA wheel로 실행할 수 있습니다.

Python은 3.10 이상을 사용합니다.

```powershell
python --version
```

## 2. 저장소 폴더로 이동

```powershell
cd C:\Users\jeyej\Desktop\chairwomans-test
```

## 3. 가상환경 생성

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

PowerShell 실행 정책 오류가 나면 현재 사용자에 한해 다음을 한 번 실행합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

그 후 다시 가상환경을 활성화합니다.

## 4. CUDA용 PyTorch 설치

PyTorch 공식 설치 페이지에서 Windows / Pip / Python / CUDA를 선택해 현재 제공되는 명령을 확인하는 것이 가장 안전합니다: <https://pytorch.org/get-started/locally/>

현재 예시 명령은 다음과 같습니다.

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

공식 페이지가 다른 CUDA wheel을 제시하면 공식 페이지의 명령을 우선 사용합니다. RTX 3070은 CUDA-capable NVIDIA GPU이므로 CPU용 PyTorch를 설치하면 안 됩니다.

## 5. 나머지 패키지 설치

```powershell
python -m pip install -r requirements.txt
```

`datasets` 패키지도 설치되며, 로컬 데이터가 없을 때 Hugging Face fallback에 사용됩니다.

GPU 인식 확인:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

정상 결과에는 `cuda available: True`와 `NVIDIA GeForce RTX 3070`이 포함되어야 합니다.

## 6. 데이터셋 폴더 준비

데이터셋 루트 하나를 만들고, 그 안에 아래 폴더를 둡니다. 폴더명은 [data/dataset_registry.py](data/dataset_registry.py)의 설정과 일치해야 합니다.

```text
D:\datasets\
├─ ImageNet\val\<class>\*.JPEG
├─ imagenet-a\images\<class>\*.JPEG
├─ imagenet-r\images\<class>\*.JPEG
├─ imagenetv2-matched-frequency-format-val\images\<class>\*.JPEG
├─ ImageNet-Sketch\images\<class>\*.JPEG
├─ DTD\<class>\*.jpg
├─ Flower102\<class>\*.jpg
├─ Food101\<class>\*.jpg
├─ StanfordCars\<class>\*.jpg
├─ SUN397\<class>\*.jpg
├─ fgvc_aircraft\<class>\*.jpg
├─ OxfordPets\<class>\*.jpg
├─ Caltech101\<class>\*.jpg
├─ UCF101\<class>\*.jpg
└─ eurosat\<class>\*.jpg
```

각 `<class>` 폴더가 한 개의 분류 클래스입니다. `ImageNet` 변형은 데이터 배포본에 따라 `images`, `test`, `val` 중 하나를 사용할 수 있으며 loader가 자동으로 찾습니다.

## 7. beta calibration

### 필수 실행 규칙

최종 test set을 실행하기 전에 반드시 `configs/betas.json`에 현재 데이터셋과 백본 조합의 beta가 존재해야 합니다.

```text
beta 존재함     -> 전체 test set 실행
beta 없음       -> 먼저 calibrate_beta.py 실행 -> beta 저장 확인 -> test 실행
```

`--beta auto`는 값이 없을 때 임의의 beta를 사용하지 않고 오류를 발생시킵니다. calibration을 건너뛴 상태로 최종 성능을 측정하지 않도록 하기 위한 동작입니다.

전체 test set을 평가하므로 이미지 샘플링용 seed는 없습니다. 다만 GMM 초기화가 완전히 재현되도록 `--seed`는 기본값 42로 유지합니다.

beta는 데이터셋·백본 조합마다 calibration split에서 grid search로 선택합니다. 기본 grid는 `0.0, 0.2, 0.35, 0.5`이며 결과는 `configs/betas.json`에 저장됩니다. calibration root는 최종 test set과 분리된 train/validation 폴더여야 합니다.

예를 들어 `D:\calibration\eurosat\train\<class>\...`가 있다면:

```powershell
python calibrate_beta.py `
  --calibration-root D:\calibration `
  --dataset eurosat `
  --backbone "ViT-B/16" `
  --split train
```

실행이 끝나면 `configs/betas.json`에 해당 데이터셋·백본의 beta가 저장됩니다. 저장된 값을 확인한 후 최종 test set을 실행합니다. `--beta auto`가 이 파일에서 beta를 자동으로 읽습니다.

```powershell
python run_comparison.py `
  --data-root D:\datasets `
  --dataset eurosat `
  --backbone "ViT-B/16" `
  --beta auto
```

calibration split이 아직 없으면 먼저 준비해야 합니다. 논문용 결과나 최종 실험에서는 `--beta 0.35`를 임의로 직접 지정하지 말고, 반드시 calibration 후 `--beta auto`를 사용합니다. `--beta 0.35`는 임시 smoke test에만 사용합니다.

### 데이터셋·백본 조합별 실행

백본이 달라지면 beta도 새로 calibration해야 합니다.

```powershell
python calibrate_beta.py --calibration-root D:\calibration --dataset eurosat --backbone "ViT-B/16" --split train
python run_comparison.py --data-root D:\datasets --dataset eurosat --backbone "ViT-B/16" --beta auto

python calibrate_beta.py --calibration-root D:\calibration --dataset eurosat --backbone "ViT-B/32" --split train
python run_comparison.py --data-root D:\datasets --dataset eurosat --backbone "ViT-B/32" --beta auto
```

`eurosat + ViT-B/16`의 beta가 존재한다고 해서 `eurosat + ViT-B/32`를 바로 실행하면 안 됩니다. 각 데이터셋·백본 조합별로 beta 존재 여부를 먼저 확인합니다.

## 8. 로컬 데이터가 없을 때 Hugging Face fallback

로컬 폴더가 있으면 로컬 데이터를 먼저 사용합니다. 로컬 폴더가 없으면 현재 기본 fallback이 등록된 데이터셋은 Hugging Face에서 자동으로 다운로드합니다.

예를 들어 ImageNet-R은 다음과 같은 설정으로 실행됩니다.

```powershell
python run_comparison.py `
  --data-root D:\datasets `
  --dataset R `
  --backbone "ViT-B/16" `
  --beta 0.35 `
  --hf-path axiong/imagenet-r `
  --hf-split test `
  --max-test-samples 20
```

`R`은 코드에 기본 등록되어 있어 `--hf-path`와 `--hf-split`을 생략해도 됩니다.

```powershell
python run_comparison.py --data-root D:\datasets --dataset R --backbone "ViT-B/16" --beta 0.35 --max-test-samples 20
```

첫 실행은 Hugging Face 캐시에 데이터를 다운로드하므로 인터넷 연결이 필요합니다. 데이터셋별 Hugging Face 저장소의 column 이름이 다르면 다음 옵션으로 지정합니다.

```powershell
python run_comparison.py `
  --data-root D:\datasets `
  --dataset eurosat `
  --hf-path <namespace>/<dataset> `
  --hf-split test `
  --hf-image-column image `
  --hf-label-column label
```

Hugging Face의 `load_dataset(path, split=...)`이 지정된 split 하나를 반환하는 방식에 맞춘 fallback입니다. [Hugging Face load_dataset 문서](https://huggingface.co/docs/datasets/package_reference/loading_methods)를 참고하세요.

## 9. 짧은 smoke test

먼저 데이터셋 하나에서 20장만 실행합니다. 첫 실행에서는 CLIP `ViT-B/16` 가중치를 자동으로 내려받으므로 인터넷 연결이 필요합니다.

```powershell
python run_comparison.py --data-root D:\datasets --dataset eurosat --backbone "ViT-B/16" --beta 0.35 --max-test-samples 20
```

확인할 내용:

- `cuda available: True`
- `Selected dataset: eurosat`
- CLIP 모델 로드 성공
- 마지막에 `CLIP`, `FINAL`, `gain` 결과 출력

## 10. 데이터셋 하나씩 전체 실행

```powershell
python run_comparison.py --data-root D:\datasets --dataset eurosat --backbone "ViT-B/16" --beta auto --seed 42
```

`--max-test-samples`를 생략하면 선택한 데이터셋의 모든 test 이미지를 평가합니다. 데이터셋 ID는 한 번에 하나만 지정합니다.

```powershell
python run_comparison.py --data-root D:\datasets --dataset I --backbone "ViT-B/16" --beta auto
python run_comparison.py --data-root D:\datasets --dataset Flower102 --backbone "ViT-B/32" --beta auto
python run_comparison.py --data-root D:\datasets --dataset eurosat --backbone "ViT-L/14" --beta auto
```

같은 작업은 제공된 PowerShell 스크립트로도 실행할 수 있습니다.

```powershell
.\scripts\run_one.ps1 -DataRoot D:\datasets -Dataset eurosat -Backbone "ViT-B/16"
```

## 11. RTX 3070에서 문제가 생길 때

- CUDA가 `False`이면 가상환경이 활성화되었는지 확인하고 CUDA용 PyTorch를 다시 설치합니다.
- `OutOfMemoryError`가 나면 먼저 `ViT-B/16`을 유지하고 smoke test의 샘플 수를 줄입니다. 이 파이프라인은 이미지 임베딩을 작은 batch로 처리합니다.
- 노트북이 멈춘 것처럼 보이면 터미널 실행을 사용해 진행률을 확인합니다.
- 처음부터 전체 실행하지 말고 `eurosat --max-test-samples 20` → 데이터셋 하나 전체 순서로 진행합니다.

## 실행 구조

실제 비교 알고리즘은 [pipeline/comparison.py](pipeline/comparison.py), 데이터셋 선택은 [data/dataset_registry.py](data/dataset_registry.py), 로컬 ImageFolder 로딩은 [data/local_datasets.py](data/local_datasets.py)에 있습니다. 실행은 `run_comparison.py` 하나로 가능하며, `MTA-main.zip`은 폴더 구조를 참고하기 위한 원본 보관본입니다.
