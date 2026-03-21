import logging
import re
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.settings import settings
from app.schemas.finetuned import DecisionEnum

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit")

_FORCE_CPU = settings.FORCE_CPU

SYSTEM_PROMPT = (
    "You are a biomedical research assistant specialized in answering questions "
    "based on biomedical evidence and hospital protocols.\n\n"
    "STRICT GUIDELINES — you must always follow these rules:\n"
    "1. Answer using the provided context only.\n"
    "2. NEVER prescribe medications, treatments, or dosages directly.\n"
    "3. Always state that clinical decisions require validation by a licensed healthcare professional.\n"
    "4. If uncertain, clearly state the limitation.\n"
    "5. Your responses are for decision support only and do not replace medical judgment."
)


class ModelManager:
    def __init__(self):
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_name: str = ""

        if _FORCE_CPU:
            self._device = "cpu"
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Modelo fine-tuned configurado para rodar em: %s", self._device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        source = self._resolve_model_source(settings.HF_MODEL_ID, settings.MODEL_PATH)
        logger.info("Carregando modelo fine-tuned de: %s", source)

        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        hf_token = getattr(settings, "HF_TOKEN", "") or None

        if hf_token:
            logger.info("HF_TOKEN detectado. Carregamento autenticado no Hugging Face.")
        else:
            logger.warning(
                "HF_TOKEN nao configurado. O modelo sera carregado sem autenticacao no Hugging Face."
            )

        tokenizer_kwargs = {
            "pretrained_model_name_or_path": source,
            "trust_remote_code": True,
        }

        model_kwargs = {
            "pretrained_model_name_or_path": source,
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }

        if self._device == "cuda":
            model_kwargs["device_map"] = "auto"

        if hf_token:
            tokenizer_kwargs["token"] = hf_token
            model_kwargs["token"] = hf_token

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        except TypeError:
            logger.warning(
                "Sua versao de transformers pode nao aceitar 'token'. Tentando com 'use_auth_token'."
            )

            tokenizer_kwargs.pop("token", None)
            model_kwargs.pop("token", None)

            if hf_token:
                tokenizer_kwargs["use_auth_token"] = hf_token
                model_kwargs["use_auth_token"] = hf_token

            self._tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        if self._device == "cpu":
            self._model = self._model.to(self._device)

        self._model.eval()
        self._model_name = source
        logger.info("Modelo fine-tuned carregado com sucesso em %s", self._device)

    def _resolve_model_source(self, hf_model_id: str, model_path: str) -> str:
        if hf_model_id:
            logger.info("HF_MODEL_ID definido. Carregando do Hugging Face: %s", hf_model_id)
            return hf_model_id

        local_path = Path(model_path)
        if model_path and local_path.exists() and any(local_path.glob("*.safetensors")):
            logger.info("Modelo local encontrado em: %s", local_path)
            return str(local_path)

        if model_path:
            logger.info("Tentando MODEL_PATH como identificador remoto: %s", model_path)
            return model_path

        raise ValueError(
            "Nenhum modelo fine-tuned configurado. Defina HF_MODEL_ID ou MODEL_PATH no .env."
        )

    def predict(
        self,
        question: str,
        context: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Modelo nao carregado. Chame load() primeiro.")

        started_at = time.time()

        user_content = self._build_user_prompt(question, context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                do_sample=settings.TEMPERATURE > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        decision = self._extract_decision(answer)
        elapsed_ms = round((time.time() - started_at) * 1000)

        if self._device == "cuda":
            torch.cuda.empty_cache()

        audit_logger.info(
            "INFERENCE | request_id=%s | provider=finetuned | decision=%s | elapsed_ms=%d | model=%s | device=%s",
            request_id or "n/a",
            decision.value,
            elapsed_ms,
            self._model_name,
            self._device,
        )

        return {
            "answer": answer,
            "decision": decision.value,
            "elapsed_ms": elapsed_ms,
            "model_name": self._model_name,
        }

    def _build_user_prompt(self, question: str, context: Optional[str]) -> str:
        if context:
            return f"Context:\n{context}\n\nQuestion: {question}"
        return f"Question: {question}"

    def _extract_decision(self, answer: str) -> DecisionEnum:
        lower = answer.lower().strip()

        for label in ("yes", "no", "maybe"):
            if re.match(rf"^{label}\b", lower):
                return DecisionEnum(label)

        for label in ("yes", "no", "maybe"):
            if re.search(rf"\b{label}\b", lower):
                return DecisionEnum(label)

        return DecisionEnum.unknown


_manager = ModelManager()


def get_model_manager() -> ModelManager:
    return _manager