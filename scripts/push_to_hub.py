"""
Publicacao do modelo fine-tuned no Hugging Face Hub.

Este script:
1. Carrega o modelo fine-tuned salvo localmente
2. Publica o modelo merged (16-bit) no Hugging Face Hub
3. Publica o tokenizer
4. Cria o model card automaticamente

Pre-requisitos:
    python scripts/train.py  (modelo deve estar treinado e salvo)

Uso:
    python scripts/push_to_hub.py

Apos publicar, o modelo estara disponivel em:
    https://huggingface.co/{HF_USERNAME}/{HF_REPO_NAME}

Para usar o modelo publicado na API sem retreinar, configure no .env:
    MODEL_PATH=<HF_USERNAME>/<HF_REPO_NAME>
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "")
HF_REPO_NAME = os.getenv("HF_REPO_NAME", "qwen35-pubmedqa")
MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "./models/qwen35-pubmedqa")
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen3.5-0.8B")

MODEL_LOCAL_PATH = ROOT_DIR / MODEL_OUTPUT_DIR.lstrip("./")
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

MAX_SEQ_LENGTH = 2048


def validate_config():
    errors = []
    if not HF_TOKEN or HF_TOKEN == "hf_SEU_TOKEN_AQUI":
        errors.append("HF_TOKEN nao configurado no .env")
    if not HF_USERNAME or HF_USERNAME == "seu_usuario":
        errors.append("HF_USERNAME nao configurado no .env")
    if not MODEL_LOCAL_PATH.exists():
        errors.append(
            f"Modelo local nao encontrado em {MODEL_LOCAL_PATH}\n"
            "Execute primeiro: python scripts/train.py"
        )
    if errors:
        for err in errors:
            print(f"Erro: {err}")
        raise SystemExit(1)


def create_model_card() -> str:
    return f"""---
language:
- en
license: apache-2.0
base_model: {BASE_MODEL_NAME}
tags:
- fine-tuned
- biomedical
- question-answering
- pubmedqa
- unsloth
- lora
datasets:
- qiaojin/PubMedQA
---

# {HF_REPO_NAME}

Fine-tuned version of [{BASE_MODEL_NAME}](https://huggingface.co/{BASE_MODEL_NAME})
on the [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) dataset
for biomedical question answering.

## Model Description

This model answers biomedical research questions with **yes**, **no**, or **maybe**,
followed by a brief explanation derived from PubMed abstract contexts.

Fine-tuned using [Unsloth](https://unsloth.ai/) with LoRA (bf16) for efficiency.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "{HF_REPO_ID}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

system_prompt = (
    "You are an expert biomedical research assistant. "
    "Given a research question and context passages from PubMed abstracts, "
    "answer with 'yes', 'no', or 'maybe', followed by a brief explanation."
)

question = "Does exercise improve cardiovascular health?"
context = "Regular aerobic exercise has been shown to reduce blood pressure..."

messages = [
    {{"role": "system", "content": system_prompt}},
    {{"role": "user", "content": f"Context:\\n{{context}}\\n\\nQuestion: {{question}}"}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

## Training Details

- **Base model**: {BASE_MODEL_NAME}
- **Dataset**: PubMedQA (PQA-L, 1000 labeled examples)
- **Method**: LoRA (bf16, r=16, alpha=16)
- **Framework**: Unsloth + TRL SFTTrainer
- **Task**: Biomedical QA (yes/no/maybe classification + explanation)

## Intended Use

Designed for biomedical research question answering. Not intended for clinical
diagnosis or medical decision-making.

## Citation

If you use this model or the PubMedQA dataset in your research, please cite
the original dataset paper:

```bibtex
@inproceedings{{jin2019pubmedqa,
  title={{PubMedQA: A Dataset for Biomedical Research Question Answering}},
  author={{Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua}},
  booktitle={{Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)}},
  pages={{2567--2577}},
  year={{2019}}
}}
```

- Paper: [arXiv:1909.06146](https://arxiv.org/abs/1909.06146)
- Dataset homepage: [https://pubmedqa.github.io/](https://pubmedqa.github.io/)
- Dataset repository: [https://github.com/pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa)
"""


def main():
    print("=" * 60)
    print("Publicando modelo no Hugging Face Hub")
    print("=" * 60)

    validate_config()

    print(f"Modelo local:  {MODEL_LOCAL_PATH}")
    print(f"Repositorio:   https://huggingface.co/{HF_REPO_ID}")
    print()

    # O modelo local ja esta merged (salvo por save_pretrained_merged em train.py),
    # entao carregamos diretamente com transformers e publicamos via push_to_hub.
    print("Carregando modelo merged do disco...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_LOCAL_PATH), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_LOCAL_PATH),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print("Publicando tokenizer e modelo no Hugging Face...")
    print("(Isso pode levar alguns minutos dependendo da conexao)")
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)

    print("Adicionando model card...")
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    card_content = create_model_card()
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Add model card",
    )

    print()
    print("Modelo publicado com sucesso!")
    print(f"URL: https://huggingface.co/{HF_REPO_ID}")
    print()
    print("Para usar o modelo publicado na API sem retreinar:")
    print(f"  Edite o .env e defina HF_MODEL_ID={HF_REPO_ID}")
    print("  Depois reinicie: docker compose up")


if __name__ == "__main__":
    main()