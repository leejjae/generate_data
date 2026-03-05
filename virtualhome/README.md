# VirtualHome Dataset Pipeline

VirtualHome 데이터셋을 준비하고 학습용 trajectory 데이터를 생성하는 전체 파이프라인입니다.

```
[1] Raw 데이터 다운로드       [2] Unity 시뮬레이터 설치      [3] Trajectory 데이터 생성
download_dataset.sh    →    download_sim.sh         →    generate_virtualhome_data.py
```

---

## 디렉토리 구조

```
virtualhome/
├── dataset/                  # raw program 데이터 (Step 1에서 생성)
├── simulation/               # Unity 바이너리 (Step 2에서 생성)
│   └── linux_exec.v2.3.0.x86_64
├── helper_scripts/
│   ├── download_dataset.sh
│   └── download_sim.sh
└── ...

scripts/
└── generate_virtualhome_data.py   # Step 3 스크립트
```

---

## Step 1: Raw 데이터 다운로드

`programs_processed_precond_nograb_morepreconds` 데이터셋을 다운로드합니다.

> **주의:** 이 스크립트는 `virtualhome/` 디렉토리 안에 `dataset/` 폴더가 존재한다고 가정합니다.
> 반드시 `virtualhome/` 디렉토리에서 실행하세요.

```bash
# dataset/ 폴더가 없으면 먼저 생성
mkdir -p virtualhome/dataset

# virtualhome/ 디렉토리에서 실행
cd virtualhome
bash helper_scripts/download_dataset.sh
```

실행 후 `virtualhome/dataset/` 안에 프로그램 파일들이 생성됩니다.

---

## Step 2: Unity 시뮬레이터 설치

VirtualHome Unity 바이너리를 다운로드합니다. OS(Linux/macOS)를 자동으로 감지하여 해당 버전을 받습니다.

```bash
# simulation/ 폴더가 없으면 먼저 생성
mkdir -p virtualhome/simulation

# 어느 디렉토리에서 실행해도 됩니다
bash virtualhome/helper_scripts/download_sim.sh
```

실행 후 `virtualhome/simulation/` 안에 바이너리가 설치됩니다.

- Linux: `linux_exec.v2.3.0.x86_64`
- macOS: `mac_exec` (또는 동등한 바이너리)

---

## Step 3: Trajectory 데이터 생성

`generate_virtualhome_data.py`는 Unity 시뮬레이터에서 `ExpertPolicy`를 실행하여 학습용 JSONL trajectory 파일을 생성합니다.

### 3-1. 시뮬레이터 먼저 실행

스크립트 실행 전, Unity 시뮬레이터를 백그라운드에서 먼저 기동해야 합니다.

```bash
# 가상 디스플레이 준비 (헤드리스 환경)
Xvfb :1 -screen 0 1024x768x24 &

# 시뮬레이터 실행 (포트 8080)
DISPLAY=:1 nohup ./virtualhome/simulation/linux_exec.v2.3.0.x86_64 \
    -batchmode -port 8080 -force-opengl \
    > /tmp/vh_sim.log 2>&1 &
```

시뮬레이터가 준비될 때까지 기다립니다.

```bash
# Player.log에 "Waiting for request"가 출력되면 준비 완료
tail -f ~/.config/unity3d/VirtualHome/VirtualHome/Player.log | grep "Waiting for request"
```

### 3-2. 데이터 생성 스크립트 실행

프로젝트 루트(`/workspace/ours/`)에서 실행합니다.

```bash
# 전체 데이터 생성 (78 tasks × 20 scenes)
python -m scripts.generate_virtualhome_data \
    --output_dir data/virtualhome_trajectories

# 특정 task만 생성
python -m scripts.generate_virtualhome_data \
    --output_dir data/virtualhome_trajectories \
    --task_ids 0 1 2

# 특정 환경(house)만 생성
python -m scripts.generate_virtualhome_data \
    --output_dir data/virtualhome_trajectories \
    --env_ids 0 1 5 6

# 시뮬레이터가 다른 포트/호스트에 있는 경우
python -m scripts.generate_virtualhome_data \
    --output_dir data/virtualhome_trajectories \
    --port 8080 --url localhost
```

### 3-3. 인수 정리

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--output_dir`, `-o` | `trajectories` | trajectory 저장 디렉토리 |
| `--port`, `-p` | `8080` | 시뮬레이터 포트 |
| `--url` | `localhost` | 시뮬레이터 호스트 |
| `--max_steps` | `100` | 에피소드당 최대 스텝 수 |
| `--task_ids` | 전체 78개 | 생성할 task ID 목록 |
| `--env_ids` | 전체 20개 | 생성할 house ID 목록 |
| `--quiet` | False | 로그 출력 억제 |

---

## 출력 형식 (JSONL)

생성된 파일은 task별 디렉토리에 house ID별 JSONL로 저장됩니다.

```
data/virtualhome_trajectories/
├── Turn_on_tv/
│   ├── env00.jsonl
│   ├── env01.jsonl
│   └── ...
├── Open_fridge/
│   └── ...
└── ...
```

각 JSONL 파일의 한 줄(한 스텝)은 다음 구조를 갖습니다.

```json
{
    "env_id": 0,
    "task_id": 3,
    "instruction": "Turn on tv",
    "action": "walk tv",
    "position_graph": {"nodes": [...], "edges": [...]},
    "visible_graph":  {"nodes": [...], "edges": [...]},
    "agent_graph":    {"nodes": [...], "edges": [...]},
    "next_visible_graph": {"nodes": [...], "edges": [...]},
    "next_agent_graph":   {"nodes": [...], "edges": [...]}
}
```

---

## 유효 house ID

TMoW 논문에서 사용된 20개의 house (seen 10개 + unseen 10개):

```
0, 1, 5, 6, 7, 8, 9, 12, 13, 15, 18, 20, 22, 24, 26, 28, 29, 31, 32, 34
```

---

## 재개(resume) 지원

이미 생성된 파일은 건너뜁니다. 중간에 중단되어도 동일한 명령으로 재실행하면 이어서 생성됩니다.

```
[42/1560] Skip Turn on tv / env0 (exists)
[43/1560] Generating: Turn on tv / env1
    Saved 3 steps → data/virtualhome_trajectories/Turn_on_tv/env01.jsonl
```

---

## 시뮬레이터 타임아웃 처리

시뮬레이터가 응답하지 않으면 자동으로 재시작 후 재시도합니다. 재시작 대기 시간은 최대 5분입니다.
