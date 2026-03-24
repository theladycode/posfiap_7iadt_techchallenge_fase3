# FIAP - Tech Challenge Fase 3: Medical Assistant API

API de suporte à decisão médica e pesquisa biomédica que combina LLMs (OpenAI e modelo fine-tuned) com RAG (Retrieval-Augmented Generation) para responder perguntas clínicas baseadas em protocolos hospitalares e contexto do paciente.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Endpoints da API](#endpoints-da-api)
- [Fluxo de Processamento](#fluxo-de-processamento)
- [Modelos de LLM](#modelos-de-llm)
- [Pipeline RAG](#pipeline-rag)
- [Auditoria](#auditoria)
- [Configuração](#configuração)
- [Como Executar](#como-executar)
- [Fine-tuning do Modelo](#fine-tuning-do-modelo)
- [Exemplos de Uso](#exemplos-de-uso)

---

## Visão Geral

O sistema funciona como um assistente médico inteligente que:

- Recebe perguntas clínicas de profissionais de saúde
- Enriquece a resposta com contexto do paciente (mock) e protocolos hospitalares via RAG
- Roteia a inferência para OpenAI GPT-4o-mini, modelo fine-tuned local (Qwen3.5-0.8B) ou modo híbrido
- Detecta níveis de risco e sinaliza questões que exigem validação humana
- Registra toda a execução em logs de auditoria com métricas detalhadas

**Stack principal:**

| Componente | Tecnologia |
|---|---|
| Framework Web | FastAPI 0.135.1 |
| Orquestração LLM | LangGraph 1.1.3 + LangChain 1.2.13 |
| LLM Principal | OpenAI GPT-4o-mini |
| Modelo Fine-tuned | Qwen/Qwen3.5-0.8B (LoRA via Unsloth) |
| Embeddings / RAG | OpenAI Embeddings + FAISS CPU |
| ML Runtime | PyTorch 2.10.0 + Transformers 5.3.0 |
| Containerização | Docker (python:3.11-slim) |

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                       FastAPI App                        │
│                    POST /assistant/query                 │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  LangGraph Workflow                      │
│                                                          │
│  load_patient_context                                    │
│         ↓                                               │
│  load_protocol_context  ◄── RAG (FAISS + OpenAI Emb.)  │
│         ↓                                               │
│  generate_llm_response  ◄── OpenAI / Fine-tuned / Both │
│         ↓                                               │
│  validate_risk                                           │
│         ↓                                               │
│  create_audit                                            │
└─────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        OpenAI         Fine-tuned     Hybrid
       GPT-4o-mini    Qwen3.5-0.8B   (ambos)
```

### Providers de LLM

| Modo (`LLM_PROVIDER`) | Comportamento |
|---|---|
| `openai` (padrão) | Usa apenas GPT-4o-mini via API |
| `finetuned` | Usa apenas Qwen3.5-0.8B local |
| `hybrid` | Fine-tuned gera contexto de suporte; GPT-4o-mini gera a resposta final |

---

## Estrutura do Projeto

```
posfiap_7iadt_techchallenge_fase3/
├── app/
│   ├── main.py                          # Entrypoint FastAPI
│   ├── api/
│   │   └── routes/
│   │       └── assistant.py             # Rotas /assistant e /health
│   ├── core/
│   │   ├── config.py                    # Configurações estáticas e system prompts
│   │   ├── settings.py                  # Variáveis de ambiente (pydantic-settings)
│   │   └── logger.py                    # Configuração de logging
│   ├── graph/
│   │   ├── workflow.py                  # Definição do grafo LangGraph
│   │   ├── state.py                     # Estado compartilhado do workflow
│   │   └── nodes.py                     # Funções de cada nó do grafo
│   ├── models/
│   │   └── model.py                     # Gerenciador do modelo fine-tuned
│   ├── rag/
│   │   ├── loader.py                    # Carregamento de documentos de protocolo
│   │   ├── splitter.py                  # Estratégia de chunking
│   │   ├── retriever.py                 # Pipeline de recuperação RAG
│   │   └── vector_store.py              # FAISS vector store
│   ├── schemas/
│   │   ├── assistant.py                 # Schemas de request/response
│   │   ├── audit.py                     # Schemas de auditoria
│   │   └── finetuned.py                 # Enum de decisão do modelo fine-tuned
│   ├── services/
│   │   ├── llm_service.py               # Roteamento de providers LLM
│   │   ├── audit_service.py             # Criação de registros de auditoria
│   │   ├── audit_repository.py          # Store in-memory de auditoria
│   │   ├── patient_service.py           # Contexto de paciente (mock)
│   │   ├── protocol_service.py          # Wrapper de recuperação de protocolos
│   │   └── providers/
│   │       ├── openai_provider.py       # Integração OpenAI (model configurável via OPENAI_MODEL_NAME)
│   │       └── finetuned_provider.py    # Inferência do modelo local
├── data/
│   ├── protocols/
│   │   ├── exames.txt                   # Protocolos de exames (hemograma, PCR)
│   │   ├── infeccao.txt                 # Protocolos de infecção e febre
│   │   └── triagem.txt                  # Protocolos de triagem
│   └── faiss_index/                     # Cache do índice FAISS (gerado automaticamente)
├── scripts/
│   ├── prepare_data.py                  # Prepara PubMedQA para fine-tuning
│   ├── train.py                         # Fine-tuning com Unsloth + LoRA
│   └── push_to_hub.py                   # Publica modelo no Hugging Face Hub
├── Dockerfile
├── requirements.txt                     # Dependências de runtime
├── requirements-train.txt               # Dependências de treinamento
├── runtime.txt                          # Versão Python (Heroku)
└── FIAP - FASE 3.postman_collection.json
```

---

## Endpoints da API

### `GET /health`

Verificação de saúde do serviço.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /assistant/query`

Endpoint principal. Envia uma pergunta clínica e recebe resposta fundamentada em protocolos.

**Request Body:**
```json
{
  "patient_id": "123",
  "question": "Quais exames solicitar para suspeita de infecção bacteriana?",
  "user_role": "doctor",
  "context": "Contexto adicional opcional"
}
```

| Campo | Tipo | Obrigatório | Validação | Descrição |
|---|---|---|---|---|
| `patient_id` | string | Sim | máx. 100 chars | ID do paciente |
| `question` | string | Sim | 1–2000 chars | Pergunta clínica |
| `user_role` | string | Sim | máx. 100 chars | Papel do usuário (ex: `doctor`, `nurse`) |
| `context` | string | Não | máx. 5000 chars | Contexto adicional |

**Response:**
```json
{
  "answer": "Com base nos protocolos hospitalares...",
  "sources": ["exames.txt", "infeccao.txt"],
  "risk_level": "high",
  "requires_human_validation": true,
  "audit": {
    "request_id": "uuid-xxxx",
    "status": "success",
    "execution_mode": "openai",
    "fallback_used": false,
    "duration_ms": 1234,
    "llm_provider": "openai",
    "model_name": "gpt-4o-mini",
    "supporting_decision": null,
    "protocol_sources": ["exames.txt"]
  }
}
```

---

### `GET /assistant/audits/{request_id}`

Recupera o log de auditoria completo de uma requisição.

**Path Parameter:** `request_id` — UUID retornado na resposta do `/assistant/query`

**Response:** Objeto `AuditDetailResponse` com todos os campos de execução, contextos utilizados, tempos e metadados do modelo.

---

## Fluxo de Processamento

O workflow é implementado com **LangGraph** como uma máquina de estados:

```
1. load_patient_context
   └─ Recupera contexto do paciente via patient_service (mock)

2. load_protocol_context
   └─ Executa busca RAG nos protocolos hospitalares (FAISS + embeddings OpenAI)

3. generate_llm_response
   └─ Roteia para o provider configurado (openai / finetuned / hybrid)
   └─ Compõe prompt com: contexto do paciente + protocolos + pergunta

4. validate_risk
   └─ Detecta palavras-chave de alto risco (prescrição, dosagem, cirurgia, etc.)
   └─ Define risk_level (high/low) e requires_human_validation

5. create_audit
   └─ Persiste log completo da execução in-memory
```

---

## Modelos de LLM

### OpenAI GPT-4o-mini

- Provider padrão
- Inferência via API (requer `OPENAI_API_KEY`)
- Temperature: `0.2`
- System prompt em português com diretrizes médicas estritas (sem prescrições diretas)

### Qwen3.5-0.8B (Fine-tuned)

- Modelo base leve (0.8B parâmetros)
- Treinado em PubMedQA com LoRA via Unsloth
- Saída: classificação `yes` / `no` / `maybe` + explicação
- Carregamento lazy (sob demanda)
- Suporte a CUDA (bf16) ou CPU (float32)
- Configurável via `HF_MODEL_ID` ou `MODEL_PATH`

### Modo Hybrid

- Fine-tuned gera uma decisão de suporte (`yes/no/maybe`)
- Essa decisão é injetada no prompt do GPT-4o-mini como contexto adicional
- Resultado final é gerado pelo GPT-4o-mini com suporte do modelo local

---

## Pipeline RAG

Documentos de protocolo são processados e indexados para recuperação semântica:

```
data/protocols/*.txt
       ↓
  DocumentLoader (encoding UTF-8 / Latin-1)
       ↓
  TextSplitter (chunk_size e overlap configuráveis via RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP)
       ↓
  OpenAI Embeddings
       ↓
  FAISS VectorStore (CPU)
       ↓
  Salvo em disco → data/faiss_index/   ← carregado nas próximas inicializações
       ↓
  Retriever (busca por similaridade)
```

**Protocolos disponíveis:**

| Arquivo | Conteúdo |
|---|---|
| `exames.txt` | Hemograma, PCR e outros exames laboratoriais |
| `infeccao.txt` | Protocolos de febre e infecção |
| `triagem.txt` | Triagem de pacientes |

Para adicionar novos protocolos, basta incluir arquivos `.txt` em `data/protocols/`.

---

## Auditoria

Toda requisição gera um registro completo de auditoria armazenado in-memory:

```json
{
  "request_id": "uuid",
  "timestamp": "2025-01-01T00:00:00Z",
  "patient_id": "123",
  "question": "...",
  "user_role": "doctor",
  "context_used": {
    "user_context": "...",
    "patient_context": "...",
    "protocol_context": "..."
  },
  "answer": "...",
  "sources": [...],
  "protocol_sources": [...],
  "risk_level": "high",
  "requires_human_validation": true,
  "llm_provider": "openai",
  "model_name": "gpt-4o-mini",
  "supporting_model_output": null,
  "supporting_decision": null,
  "status": "success",
  "execution_mode": "openai",
  "fallback_used": false,
  "started_at": "...",
  "finished_at": "...",
  "duration_ms": 1200,
  "error_message": null
}
```

> **Nota:** O armazenamento é in-memory e não persiste entre reinicializações. Para produção, integre um banco de dados.

---

## Configuração

Crie um arquivo `.env` na raiz do projeto:

```env
# Obrigatório
OPENAI_API_KEY=sk-...

# Provider LLM: openai | finetuned | hybrid
LLM_PROVIDER=openai
OPENAI_MODEL_NAME=gpt-4o-mini

# Modelo fine-tuned (necessário se LLM_PROVIDER != openai)
HF_MODEL_ID=                          # ID do modelo no Hugging Face Hub
MODEL_PATH=                           # Caminho local alternativo
HF_TOKEN=                             # Token HF (para modelos privados)
BASE_MODEL_NAME=Qwen/Qwen3.5-0.8B

# Parâmetros de geração
MAX_NEW_TOKENS=256
TEMPERATURE=0.1
TOP_P=0.9
MAX_SEQ_LENGTH=2048

# RAG
PROTOCOLS_PATH=data/protocols
RAG_CHUNK_SIZE=300
RAG_CHUNK_OVERLAP=50
FAISS_INDEX_PATH=data/faiss_index

# Runtime
ENV=dev
DEBUG=True
FORCE_CPU=True
```

---

## Como Executar

### Localmente

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar variáveis de ambiente
cp .env.example .env  # edite com suas chaves

# 3. Iniciar a API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

A documentação interativa estará disponível em `http://localhost:8000/docs`.

### Docker

```bash
# Build da imagem
docker build -t medical-assistant:latest .

# Execução
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="sk-..." \
  -e LLM_PROVIDER="openai" \
  medical-assistant:latest
```

---

## Fine-tuning do Modelo

Os scripts em `scripts/` permitem treinar e publicar o modelo fine-tuned.

### 1. Preparar os dados

```bash
python scripts/prepare_data.py
```

Baixa e formata o dataset **PubMedQA** (1000 exemplos rotulados) no formato ChatML compatível com Qwen3.5.

### 2. Treinar

```bash
pip install -r requirements-train.txt
python scripts/train.py
```

**Configuração de treinamento (LoRA via Unsloth):**

| Parâmetro | Valor |
|---|---|
| Base model | Qwen/Qwen3.5-0.8B |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Learning rate | 2e-4 |
| Batch size | 4 |
| Max steps | 300 |
| Warmup steps | 10 |
| Avaliação | A cada 50 steps |
| Checkpoints | A cada 100 steps |
| Precisão | bf16 (CUDA) |

### 3. Publicar no Hugging Face Hub

```bash
python scripts/push_to_hub.py
```

Após publicar, configure `HF_MODEL_ID` com o ID do repositório e `LLM_PROVIDER=finetuned` (ou `hybrid`).

---

## Exemplos de Uso

### Consulta simples (modo OpenAI)

```bash
curl -X POST http://localhost:8000/assistant/query \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "question": "Quais exames solicitar para suspeita de infecção bacteriana?",
    "user_role": "doctor"
  }'
```

### Consulta com contexto adicional

```bash
curl -X POST http://localhost:8000/assistant/query \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P002",
    "question": "O exercício aeróbico reduz risco cardiovascular?",
    "user_role": "nurse",
    "context": "Meta-análise de 30 RCTs demonstrou redução da pressão arterial com exercício regular."
  }'
```

### Recuperar auditoria

```bash
curl http://localhost:8000/assistant/audits/{request_id}
```

> A coleção Postman completa está disponível em `FIAP - FASE 3.postman_collection.json`.

---

## Notas Importantes

- O sistema **não realiza prescrições diretas** — respostas de alto risco são sinalizadas com `requires_human_validation: true`
- O contexto de paciente atual é **mock** — integração com sistemas reais (HIS/EHR) deve ser implementada em `patient_service.py`
- O armazenamento de auditoria é **in-memory** — para produção, substitua `audit_repository.py` por uma solução persistente
- Para usar o modelo fine-tuned localmente, é necessário hardware compatível (GPU recomendada; CPU suportado via `FORCE_CPU=True`)
