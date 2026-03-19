Medical Assistant API (AI + RAG + LangGraph)

Este projeto implementa um **assistente médico virtual inteligente**, capaz de auxiliar profissionais de saúde com base em:

- Protocolos internos do hospital
- Contexto do paciente
- Modelos de linguagem (LLM)
- Fluxos de decisão estruturados (LangGraph)
- Recuperação de informação com RAG (Retrieval-Augmented Generation)

---

## Objetivo

Simular um sistema hospitalar avançado capaz de:

- Responder dúvidas clínicas
- Sugerir condutas com base em protocolos
- Identificar riscos nas respostas
- Garantir rastreabilidade (auditoria)
- Utilizar dados contextuais do paciente

---

## Arquitetura

O sistema utiliza um **workflow baseado em LangGraph**, estruturando o fluxo de decisão:

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