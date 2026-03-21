"""
Fine-tuning do Qwen3.5-0.8B com dados PubMedQA usando Unsloth.

Este script realiza:
1. Carregamento do modelo base Qwen3.5-0.8B via Unsloth
2. Aplicacao de LoRA (bf16 pois o Qwen3.5 nao suporta QLoRA 4-bit)
3. Treinamento supervisionado (SFT) com os dados PubMedQA formatados
4. Avaliacao no conjunto de teste
5. Salvamento do modelo fine-tuned localmente
6. Publicacao automatica no Hugging Face (se HF_TOKEN e HF_USERNAME estiverem no .env)

Pre-requisitos:
    python scripts/prepare_data.py  (deve ser executado antes)

Uso:
    python scripts/train.py

VRAM necessaria: ~3GB (LoRA bf16 no Qwen3.5-0.8B)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen3.5-0.8B")
MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "./models/qwen35-pubmedqa")
MODEL_OUTPUT_PATH = ROOT_DIR / MODEL_OUTPUT_DIR.lstrip("./")

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "")
HF_REPO_NAME = os.getenv("HF_REPO_NAME", "qwen35-pubmedqa")

MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
# usando batch size maior para melhor performance
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 10
MAX_STEPS = 300
LEARNING_RATE = 2e-4
LOGGING_STEPS = 10
SEED = 3407


def load_sft_dataset(path: Path):
    """Carrega o dataset SFT formatado e retorna um HuggingFace Dataset."""
    from datasets import Dataset

    with open(path) as f:
        data = json.load(f)
    return Dataset.from_list(data)


def main():
    sft_train_path = DATA_DIR / "sft_train.json"
    sft_test_path = DATA_DIR / "sft_test.json"

    if not sft_train_path.exists():
        print("Erro: dados de treinamento nao encontrados.")
        print("Execute primeiro: python scripts/prepare_data.py")
        raise SystemExit(1)

    print("=" * 60)
    print("Fine-tuning Qwen3.5-0.8B com PubMedQA via Unsloth")
    print("=" * 60)
    print(f"Modelo base:   {BASE_MODEL_NAME}")
    print(f"Saida:         {MODEL_OUTPUT_PATH}")
    print(f"Max seq len:   {MAX_SEQ_LENGTH}")
    print(f"LoRA r:        {LORA_R}")
    print(f"Max steps:     {MAX_STEPS}")
    print()

    print("Carregando modelo e tokenizer...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    print("Aplicando LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("Carregando datasets...")
    train_dataset = load_sft_dataset(sft_train_path)
    eval_dataset = load_sft_dataset(sft_test_path)
    print(f"  Treino: {len(train_dataset)} exemplos")
    print(f"  Teste:  {len(eval_dataset)} exemplos")

    from trl import SFTConfig, SFTTrainer
    from unsloth import unsloth_train  # usado como fallback se GA > 1

    outputs_dir = ROOT_DIR / "outputs_qwen35"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=LOGGING_STEPS,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            output_dir=str(outputs_dir),
            optim="adamw_8bit",
            seed=SEED,
            dataset_num_proc=2,
            bf16=True,
            report_to="none",
        ),
    )

    print("\nIniciando treinamento...")
    # Size batch comportou GPU (5070 - 12gb de VRAM). Usando gradient accumulation para melhor performance.
    # Referencia: https://unsloth.ai/blog/gradient
    if GRADIENT_ACCUMULATION_STEPS > 1:
        trainer_stats = unsloth_train(trainer)
    else:
        trainer_stats = trainer.train()

    print("\nTreinamento concluido!")
    print(f"  Steps:          {trainer_stats.global_step}")
    print(f"  Loss final:     {trainer_stats.training_loss:.4f}")
    elapsed = trainer_stats.metrics.get("train_runtime", 0)
    print(f"  Tempo total:    {elapsed:.1f}s")

    print(f"\nSalvando modelo merged (16-bit) em {MODEL_OUTPUT_PATH} ...")
    MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Salva o modelo fundido (LoRA mergeado nos pesos base) para que a API
    # possa carrega-lo diretamente com AutoModelForCausalLM sem precisar do PEFT.
    model.save_pretrained_merged(
        str(MODEL_OUTPUT_PATH),
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"Modelo salvo em {MODEL_OUTPUT_PATH}")

    hf_repo_id = publish_to_huggingface(model, tokenizer)

    print()
    print("Proximos passos:")
    if hf_repo_id:
        print(f"  1. Defina no .env:  HF_MODEL_ID={hf_repo_id}")
    print("  2. Inicie a API:    docker compose up")


def publish_to_huggingface(model, tokenizer) -> str | None:
    """
    Publica o modelo no Hugging Face automaticamente apos o treinamento.
    Retorna o repo ID publicado, ou None se as credenciais nao estiverem configuradas.
    """
    if not HF_TOKEN or not HF_USERNAME:
        print("\nAviso: HF_TOKEN ou HF_USERNAME nao configurados no .env.")
        print("  Para publicar manualmente: python scripts/push_to_hub.py")
        return None

    hf_repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    print(f"\nPublicando modelo no Hugging Face: {hf_repo_id} ...")
    print("(Isso pode levar alguns minutos dependendo da conexao)")

    try:
        model.push_to_hub_merged(
            hf_repo_id,
            tokenizer,
            save_method="merged_16bit",
            token=HF_TOKEN,
        )
        print(f"Modelo publicado com sucesso!")
        print(f"  URL: https://huggingface.co/{hf_repo_id}")
        print(f"\nPara usar sem retreinar, defina no .env:")
        print(f"  HF_MODEL_ID={hf_repo_id}")
        return hf_repo_id
    except Exception as exc:
        print(f"Aviso: Falha ao publicar no Hugging Face: {exc}")
        print("  Para publicar manualmente: python scripts/push_to_hub.py")
        return None


if __name__ == "__main__":
    main()