"""
Preparacao dos dados PubMedQA para fine-tuning do Qwen3.5.

Este script:
1. Faz download do dataset PubMedQA (PQA-L) do Hugging Face
2. Executa a logica de split equivalente ao split_dataset.py do repositorio oficial
3. Formata os dados no estilo de instrucao para SFT com Qwen3.5 (ChatML)
4. Salva os datasets formatados em data/

Uso:
    python scripts/prepare_data.py
"""

import json
import math
import os
import random
import sys
from functools import reduce
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RANDOM_SEED = 0
random.seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

SYSTEM_PROMPT = (
    "You are a biomedical research assistant specialized in answering questions "
    "based on PubMed scientific literature.\n\n"
    "STRICT GUIDELINES — you must always follow these rules:\n"
    "1. Answer research questions with 'yes', 'no', or 'maybe', followed by a "
    "brief evidence-based explanation drawn from the provided context.\n"
    "2. NEVER prescribe medications, treatments, or dosages directly. "
    "Always state that clinical decisions require validation by a licensed "
    "healthcare professional.\n"
    "3. NEVER replace professional medical advice, diagnosis, or treatment. "
    "If the question involves patient-specific clinical decisions, explicitly "
    "recommend consulting a qualified physician.\n"
    "4. Base your answers strictly on the provided context or established "
    "biomedical research. Do not speculate beyond the evidence.\n"
    "5. When uncertain, answer 'maybe' and explain the limitations of the "
    "available evidence.\n"
    "6. Your responses are intended for research and informational purposes only."
)


def split_label(pmids: list[str], fold: int) -> list[list[str]]:
    random.shuffle(pmids)
    num_all = len(pmids)
    num_split = math.ceil(num_all / fold)
    output = []
    for i in range(fold):
        if i == fold - 1:
            output.append(pmids[i * num_split :])
        else:
            output.append(pmids[i * num_split : (i + 1) * num_split])
    return output


def split_dataset(dataset: dict, fold: int) -> list[dict]:
    add = lambda x: reduce(lambda a, b: a + b, x)

    label2pmid: dict[str, list[str]] = {"yes": [], "no": [], "maybe": []}
    for pmid, info in dataset.items():
        label2pmid[info["final_decision"]].append(pmid)

    label2pmid = {k: split_label(v, fold) for k, v in label2pmid.items()}

    output = []
    for i in range(fold):
        pmids = add([v[i] for _, v in label2pmid.items()])
        output.append({pmid: dataset[pmid] for pmid in pmids})

    if len(output[-1]) != len(output[0]):
        for i in range(fold - 1):
            pmids_list = list(output[i])
            picked = random.choice(pmids_list)
            output[-1][picked] = output[i][picked]
            output[i].pop(picked)

    return output


def combine_other(cv_sets: list[dict], fold: int) -> dict:
    output = {}
    for i in range(10):
        if i != fold:
            for pmid, info in cv_sets[i].items():
                output[pmid] = info
    return output


def format_example(pmid: str, info: dict) -> dict:
    """Converte um registro PubMedQA para formato ChatML compativel com Qwen3.5."""
    contexts = info.get("CONTEXTS", [])
    context_text = "\n\n".join(
        f"[Context {i + 1}] {ctx}" for i, ctx in enumerate(contexts)
    )
    question = info.get("QUESTION", "")
    decision = info.get("final_decision", "")
    long_answer = info.get("LONG_ANSWER", "")

    user_content = f"Context:\n{context_text}\n\nQuestion: {question}"
    assistant_content = f"{decision}\n\n{long_answer}".strip()

    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )

    return {"pmid": pmid, "text": text, "label": decision}


def download_pqal_from_huggingface() -> dict:
    """Faz download do PQA-L diretamente do Hugging Face datasets."""
    print("Baixando PubMedQA (PQA-L) do Hugging Face...")
    try:
        from datasets import load_dataset

        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        dataset = {}
        for row in ds:
            pmid = str(row["pubid"])
            dataset[pmid] = {
                "QUESTION": row["question"],
                "CONTEXTS": row["context"]["contexts"],
                "LONG_ANSWER": row["long_answer"],
                "final_decision": row["final_decision"],
                "YEAR": row.get("year", ""),
            }
        print(f"  {len(dataset)} registros carregados do PQA-L.")
        return dataset
    except Exception as e:
        print(f"Erro ao baixar do Hugging Face: {e}")
        sys.exit(1)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = download_pqal_from_huggingface()

    ori_pqal_path = DATA_DIR / "ori_pqal.json"
    with open(ori_pqal_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset original salvo em {ori_pqal_path}")

    print("\nExecutando split do dataset (equivalente ao split_dataset.py pqal)...")
    random.seed(RANDOM_SEED)

    cv_set, testset = split_dataset(dataset, 2)

    test_path = DATA_DIR / "test_set.json"
    with open(test_path, "w") as f:
        json.dump(testset, f, indent=2)
    print(f"  Test set: {len(testset)} exemplos -> {test_path}")

    cv_sets = split_dataset(cv_set, 10)
    for i in range(10):
        fold_dir = DATA_DIR / f"pqal_fold{i}"
        fold_dir.mkdir(exist_ok=True)
        with open(fold_dir / "dev_set.json", "w") as f:
            json.dump(cv_sets[i], f, indent=2)
        train_fold = combine_other(cv_sets, i)
        with open(fold_dir / "train_set.json", "w") as f:
            json.dump(train_fold, f, indent=2)
    print(f"  10 folds de CV criados em {DATA_DIR}/pqal_fold*/")

    print("\nFormatando dados para SFT (formato ChatML)...")
    train_data = cv_sets[0]
    train_examples = [format_example(pmid, info) for pmid, info in train_data.items()]
    test_examples = [format_example(pmid, info) for pmid, info in testset.items()]

    for fold_i in range(1, 10):
        for pmid, info in cv_sets[fold_i].items():
            train_examples.append(format_example(pmid, info))

    sft_train_path = DATA_DIR / "sft_train.json"
    with open(sft_train_path, "w") as f:
        json.dump(train_examples, f, indent=2)
    print(f"  SFT train: {len(train_examples)} exemplos -> {sft_train_path}")

    sft_test_path = DATA_DIR / "sft_test.json"
    with open(sft_test_path, "w") as f:
        json.dump(test_examples, f, indent=2)
    print(f"  SFT test:  {len(test_examples)} exemplos -> {sft_test_path}")

    print("\nDistribuicao de labels no treino:")
    from collections import Counter

    label_counts = Counter(ex["label"] for ex in train_examples)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print("\nPreparacao de dados concluida")


if __name__ == "__main__":
    main()