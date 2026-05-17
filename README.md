# Seonji LLM SFT

선지 생성용 LLM SFT 실험 프로젝트입니다.

## 구성

- `train_sft_colab.ipynb`: Colab 학습 런처
- `train_dpo_colab.ipynb`: SFT adapter 기반 DPO 학습 런처
- `scripts/train_sft.py`: QLoRA SFT 학습 스크립트
- `scripts/train_dpo.py`: QLoRA DPO 학습 스크립트
- `scripts/upload_hf_adapter.py`: Hugging Face Hub 업로드 스크립트
- `configs/train_config_v1.json`: 학습 설정
- `configs/train_dpo_config_v1.json`: DPO 학습 설정
- `configs/prompt_template_v1.md`: 프롬프트 템플릿
- `sft_dataset.json`: 학습 데이터

## Colab 실행 흐름

1. `train_sft_colab.ipynb`를 Colab에서 엽니다.
2. 첫 설정 셀에서 `GITHUB_REPO_URL`, `HF_USERNAME`, 실험 버전/가설을 수정합니다.
3. 위에서부터 순서대로 실행합니다.
4. 학습 결과는 Colab 임시 디스크 `/content/sft_outputs/runs`에 저장됩니다.
5. 학습 완료 후 adapter와 실험 문서는 Hugging Face Hub에 업로드됩니다.

## Hugging Face 저장 규칙

repo 이름은 다음 형식으로 자동 생성됩니다.

```text
{project_name}-{model_name}-{experiment_version}
```

예시:

```text
seonji-qwen35-4b-v1
seonji-gemma4-e4b-it-v1
seonji-exaone35-24b-instruct-v1
```

repo에는 adapter 파일과 `run_artifacts/<run_id>/` 아래의 평가 결과, 설정, 실험 노트가 함께 저장됩니다.
DPO 학습은 `test_set_v1.json`으로 최종 생성 평가를 수행하고, `test_generation_results.json`과 `test_generation_summary.json`도 같은 경로에 함께 저장합니다.
