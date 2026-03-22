# Pós Tech FIAP Fase 3

##Medical Assistant API (AI + RAG + LangGraph)

Este projeto implementa um **assistente médico virtual inteligente**, capaz de auxiliar profissionais de saúde com base em:

- Protocolos internos do hospital
- Contexto do paciente
- Modelos de linguagem (LLM)
- Fluxos de decisão estruturados (LangGraph)
- Recuperação de informação com RAG (Retrieval-Augmented Generation)
- Modelo biomédico fine-tuned (PubMedQA)

---

## Objetivo

Simular um sistema hospitalar avançado capaz de:

- Responder dúvidas clínicas
- Sugerir condutas com base em protocolos
- Identificar riscos nas respostas
- Garantir rastreabilidade (auditoria)
- Utilizar dados contextuais do paciente
- Combinar múltiplos modelos (estratégia híbrida)

---

## Arquitetura

O sistema utiliza um **workflow baseado em LangGraph**, estruturando o fluxo de decisão:
```bash
load_patient_context
        ↓
load_protocol_context
        ↓
generate_llm_response
        ↓
validate_risk
        ↓
create_audit
```

### Providers e LLM
O sistema suporta múltiplos providers:

- openai: GPT-4o-mini
- finetuned: Modelo biomédico (Qwen3.5 fine-tuned PubMedQA)
- hybrid: Combina evidência científica + resposta clínica

#### Auditoria
Cada requisição gera logs estruturados:
```bash
{
  "timestamp": "...",
  "patient_id": "123",
  "question": "...",
  "answer": "...",
  "risk_level": "high",
  "requires_human_validation": true,
  "llm_provider": "hybrid"
}
```

## 🛠️ Instalação e execução

### Pré-requisitos
1. Ter o **Python** instalado na máquina
2. Fazer o **clone do repositório**:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   ```

### Instalação
1. Criar e ativar um ambiente virtual (`venv`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Instalar as dependências do projeto:
   ```bash
   pip install -r requirements.txt
   ```
3. Executar a API
   ```bash
   uvicorn app.main:app --reload
   ```
   A API estará disponível em:
      http://localhost:8000
