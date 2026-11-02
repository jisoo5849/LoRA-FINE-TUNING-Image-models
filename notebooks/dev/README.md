# 노트북 개발 도구

`../lora_training.ipynb` 는 이 폴더의 빌더로 만든다. 노트북을 직접 고치지 말고 빌더나 어댑터를 고친 뒤 다시 만든다.

| 파일 | 하는 일 |
|---|---|
| `build_notebook.py` | 셀 원본과 `adapters/*.py` 를 합쳐 노트북을 만든다 |
| `adapters/<계열>.py` | 모델 계열별 어댑터 (규격은 `INTERFACE.md`) |
| `harness.py` | 어댑터 하나를 아주 작은 무작위 모델로 시험한다 (인코딩, 손실, 기울기, LoRA 저장·다시 불러오기, 생성) |
| `run_nb.py` | 노트북 전체를 시험 모드로 실행한다 |

```
python build_notebook.py adapters ../lora_training.ipynb
python harness.py qwen_image                 # HARNESS OK qwen_image
python run_nb.py qwen_image cpu x            # qwen_image cpu OK
python run_nb.py qwen_image cuda x           # GPU·bf16 시험
```

- 시험 버전: torch 2.15.0 (CPU 판, CUDA 13.2 판), diffusers 0.42.1, peft 0.21.2, transformers 5.22.1, accelerate 1.15.0, Python 3.14.
- 작은 토크나이저를 Hugging Face 에서 내려받으므로 시험에는 인터넷이 필요하다.
