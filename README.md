# Passos Mágicos — Predição de Defasagem Escolar

> **FIAP Pós Tech — Machine Learning Engineering | Datathon 2024**
>
> Sistema de classificação binária para identificar alunos em situação de defasagem escolar na ONG Passos Mágicos, com API REST para integração em produção.

---

## Índice

1. [Contexto do Problema](#1-contexto-do-problema)
2. [Arquitetura do Projeto](#2-arquitetura-do-projeto)
3. [Estrutura de Diretórios](#3-estrutura-de-diretórios)
4. [Pré-requisitos](#4-pré-requisitos)
5. [Instalação Local](#5-instalação-local)
6. [Instalação via Docker](#6-instalação-via-docker)
7. [Treino do Modelo](#7-treino-do-modelo)
8. [MLflow — Rastreamento de Experimentos](#8-mlflow--rastreamento-de-experimentos)
9. [Executando a API](#9-executando-a-api)
10. [Usando a API — Endpoints](#10-usando-a-api--endpoints)
11. [Exemplos de Requisição](#11-exemplos-de-requisição)
12. [Testes](#12-testes)
13. [Detalhes do Modelo](#13-detalhes-do-modelo)
14. [Decisões Técnicas Relevantes](#14-decisões-técnicas-relevantes)
15. [Monitoramento e Logs](#15-monitoramento-e-logs)
16. [CI/CD — Integração e Entrega Contínua](#16-cicd--integração-e-entrega-contínua)
17. [Resolução de Problemas](#17-resolução-de-problemas)

---

## 1. Contexto do Problema

A **Passos Mágicos** é uma ONG que transforma a vida de crianças e jovens em situação de vulnerabilidade social por meio da educação. Anualmente, a organização avalia seus alunos através do **PEDE (Pesquisa Extensiva do Desenvolvimento Educacional)**, que gera indicadores de desempenho para cada estudante.

### O que é defasagem escolar?

Um aluno é considerado **defasado** quando está matriculado em uma fase inferior à esperada para sua idade — ou seja, quando a coluna `Defas` do dataset assume valor negativo (ex: `Defas = -2` significa 2 anos abaixo do esperado).

### Objetivo do modelo

Classificar automaticamente, com base nos indicadores do PEDE, se um aluno está:

| Classe | Label | Significado |
|--------|-------|-------------|
| `1` | **defasado** | `Defas < 0` — abaixo do nível esperado |
| `0` | **no nível** | `Defas >= 0` — no nível ou adiantado |

Esse modelo permite à equipe pedagógica priorizar ações de suporte antes mesmo da avaliação formal completa, utilizando apenas os indicadores quantitativos disponíveis.

---

## 2. Arquitetura do Projeto

> **Stack tecnológica:** inclui MLflow 3.x para rastreamento de experimentos e versionamento do modelo.

```
Dataset PEDE 2024 (.xlsx)
        │
        ▼
┌─────────────────────┐
│   preprocessing.py  │  ← rename, select, convert, impute, clip
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────┐
│  feature_engineering.py │  ← pedra_num, avg_notas, eng_aprendizagem
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│            sklearn Pipeline                          │
│  SimpleImputer → StandardScaler → GradientBoosting  │
└──────────┬───────────────────────────────────────────┘
           │  GridSearchCV (cv=5, scoring=f1)
           │
           ├──────────────────────────────────────────────────────┐
           ▼                                                      ▼
     models/model.pkl                                    MLflow Tracking
     models/metrics.json                          ┌─────────────────────────┐
           │                                      │  Experimento, params,   │
           │                                      │  métricas, artefatos,   │
           │                                      │  Model Registry         │
           │                                      └──────────┬──────────────┘
           │                                                 │
           │                                        mlflow ui (porta 5000)
           ▼
┌─────────────────────┐
│    FastAPI (API)    │  ← POST /predict
│    app/main.py      │  (carrega model.pkl via joblib)
│    app/routes.py    │
└─────────────────────┘
```

**Stack tecnológica:**

| Camada | Tecnologia |
|--------|-----------|
| Linguagem | Python 3.11 |
| ML | scikit-learn 1.8, joblib |
| Rastreamento | MLflow 3.10 |
| Dados | pandas, numpy, openpyxl |
| API | FastAPI, uvicorn |
| Testes | pytest, pytest-cov |
| Deploy | Docker, Docker Compose |
| Monitoramento | Prometheus, Grafana |
| CI/CD | GitHub Actions |

---

## 3. Estrutura de Diretórios

```
passos_magicos_datathon/
│
├── data/
│   ├── BASE DE DADOS PEDE 2024 - DATATHON.xlsx   ← dataset de treino
│   └── PEDE_PASSOS_DATASET_FIAP.xlsx              ← dataset histórico
│
├── models/
│   ├── model.pkl          ← pipeline treinado (gerado por train.py)
│   └── metrics.json       ← métricas do modelo no conjunto de teste
│
├── logs/
│   ├── train.log          ← logs do treino (gerado automaticamente)
│   └── api.log            ← logs da API em produção
│
├── notebooks/
│   └── eda.ipynb          ← análise exploratória dos dados
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        ← limpeza e preparação dos dados
│   ├── feature_engineering.py  ← criação de features derivadas
│   ├── train.py                ← script principal de treino
│   ├── evaluate.py             ← métricas e relatório de avaliação
│   └── utils.py                ← logging, save/load model
│
├── app/
│   ├── __init__.py
│   ├── main.py            ← instância FastAPI + lifespan (carrega modelo)
│   └── routes.py          ← endpoints: /, /health, /predict
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_evaluate.py
│   ├── test_train.py
│   └── test_utils.py
│
├── .github/
│   └── workflows/
│       ├── ci.yml             ← pipeline de testes e build (todo push)
│       └── cd.yml             ← publicação da imagem Docker (merge na main)
│
├── Dockerfile
├── docker-compose.yml         ← sobe API + Prometheus + Grafana
├── prometheus.yml             ← configuração de scraping do Prometheus
├── requirements.txt
├── CICD_OVERVIEW.md           ← visão geral de CI/CD aplicada ao projeto
└── README.md
```

---

## 4. Pré-requisitos

### Instalação local

- Python **3.11** ou **3.12** com módulo `venv` disponível
- Acesso ao arquivo `data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx`

> **Todos os comandos** nas seções seguintes assumem que o ambiente virtual está **ativado**. Ative-o sempre antes de usar o projeto:</p>
>
> ```bash
> source .venv/bin/activate   # Linux/macOS
> .venv\Scripts\Activate.ps1  # Windows (PowerShell)
> ```

### Docker

- Docker Engine **20.10+** e Docker Compose **v2+**
- `model.pkl` deve existir em `models/` antes do build (veja [Treino do Modelo](#7-treino-do-modelo))

---

## 5. Instalação Local

### 5.1 Clone e acesse o projeto

```bash
git clone <url-do-repositorio>
cd passos_magicos_datathon
```

### 5.2 Crie e ative o ambiente virtual

```bash
# Criar o ambiente virtual (apenas uma vez)
python3 -m venv .venv

# Ativar — Linux/macOS
source .venv/bin/activate

# Ativar — Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Após ativar, o prompt do terminal exibirá `(.venv)` indicando que o ambiente está ativo. **Todos os passos seguintes devem ser executados com o venv ativado.**

Para desativar ao terminar:

```bash
deactivate
```

### 5.3 Instale as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.4 Verifique a instalação

```bash
python -c "import fastapi, sklearn, mlflow, pandas; print('OK')"
```

---

## 6. Instalação via Docker

O Docker encapsula todas as dependências e o modelo treinado, garantindo ambiente reproduzível.

> **Importante:** o `model.pkl` deve existir em `models/` antes de construir a imagem. Execute o treino local primeiro (passo 7).

### 6.1 Build da imagem

```bash
docker build -t passos-magicos .
```

### 6.2 Executar o container

```bash
docker run -p 8000:8000 passos-magicos
```

### 6.3 Com volume para logs persistentes

```bash
docker run -p 8000:8000 -v $(pwd)/logs:/app/logs passos-magicos
```

### 6.4 Em background (modo detached)

```bash
docker run -d --name passos-api -p 8000:8000 passos-magicos

# Verificar status
docker ps

# Ver logs em tempo real
docker logs -f passos-api

# Parar e remover
docker stop passos-api && docker rm passos-api
```

### 6.5 Healthcheck

O Docker verifica automaticamente o endpoint `/health` a cada 30 segundos. Você pode inspecionar o status com:

```bash
docker inspect --format='{{.State.Health.Status}}' passos-api
```

### 6.6 Stack completa com monitoramento (recomendado)

Para subir a API junto com Prometheus e Grafana em um único comando:

```bash
docker compose up --build
```

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| API FastAPI | http://localhost:8000/docs | — |
| Métricas (Prometheus) | http://localhost:8000/metrics | — |
| Prometheus UI | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin` / `admin` |

**Configurar Grafana (uma vez):**
1. Acesse http://localhost:3000 e faça login
2. **Connections > Add data source > Prometheus** → URL: `http://prometheus:9090` → Save & Test
3. **Dashboards > Import** → ID `19797` → Import (dashboard FastAPI pronto)

Para parar tudo:
```bash
docker compose down
```

---

## 7. Treino do Modelo

### 7.1 Execute o script de treino

A partir da raiz do projeto (com `.venv` ativado):

```bash
python src/train.py
```

O script executa automaticamente:

1. Carrega `data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx`
2. Cria o target binário (`Defas < 0 → 1`)
3. Pré-processa as features (renomeia, converte, imputa, clipa)
4. Gera features derivadas (pedra_num, avg_notas, eng_aprendizagem)
5. Divide em treino (80%) e teste (20%), estratificado
6. Busca os melhores hiperparâmetros com `GridSearchCV` (cv=5, scoring=F1)
7. Avalia no conjunto de teste e imprime o relatório
8. Salva `models/model.pkl` e `models/metrics.json`
9. Grava logs em `logs/train.log`

### 7.2 Saída esperada

```
==================================================
RELATÓRIO DE AVALIAÇÃO DO MODELO
==================================================
  Acurácia:  0.8895
  F1-Score:  0.9249  ← métrica principal
  AUC-ROC:   0.9245  ← qualidade geral
  Precisão:  0.8797
  Recall:    0.9750
==================================================

Matriz de Confusão:
  TN=36  FP=16
  FN=3   TP=117
==================================================
```

### 7.3 Arquivos gerados

| Arquivo | Descrição |
|---------|-----------|
| `models/model.pkl` | Pipeline sklearn serializado (imputer + scaler + modelo) |
| `models/metrics.json` | Métricas do conjunto de teste em JSON |
| `logs/train.log` | Log completo do processo de treino |

### 7.4 Retreino

Para retreinar com novos dados, basta substituir o `.xlsx` no diretório `data/` e rodar `python src/train.py` novamente (com o venv ativado). O `model.pkl` será sobrescrito.

---

## 8. MLflow — Rastreamento de Experimentos

O MLflow é integrado de forma **aditiva**: cada execução de `train.py` registra automaticamente o experimento, os hiperparâmetros, as métricas e o modelo no registry — sem alterar o comportamento existente da API ou do pipeline de treino.

### 8.1 O que é registrado automaticamente

A cada `python src/train.py`, o MLflow grava:

| Categoria | O que é registrado |
|-----------|-------------------|
| **Tags** | Dataset, algoritmo, n_samples, n_features, target |
| **Parâmetros** | Todos os hiperparâmetros do GridSearchCV + cv_folds, test_size, random_state, imputer_strategy |
| **Métricas** | f1_score, roc_auc, precision, recall, accuracy, cv_f1_best |
| **Artefatos** | `model.pkl` (joblib), `metrics.json`, modelo sklearn no formato MLflow |
| **Registry** | Modelo registrado como `PassosMagicos-DefasagemClassifier` |

### 8.2 Iniciar o MLflow UI

O projeto usa **SQLite** como backend de rastreamento (padrão recomendado no MLflow 3.x). O banco é criado automaticamente em `mlruns/mlflow.db` na primeira execução de `train.py`.

```bash
# Com o venv ativado, mlflow já estará no PATH
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 0.0.0.0 --port 5000
```

Acesse no navegador: **http://localhost:5000**

O painel exibe:
- Tabela comparativa de todos os runs (métricas, parâmetros)
- Gráficos de evolução de métricas entre runs
- Artefatos de cada experimento (modelo, métricas JSON)
- Model Registry com versões e aliases

### 8.3 Comparar experimentos via CLI

```bash
# Listar todos os runs do experimento
mlflow runs list --experiment-name passos-magicos-defasagem

# Ver detalhes de um run específico
mlflow runs describe --run-id <run_id>
```

### 8.4 Carregar modelo do MLflow Registry em Python

```python
from src.utils import load_model_from_mlflow

# Versão mais recente (padrão)
model = load_model_from_mlflow("PassosMagicos-DefasagemClassifier")

# Versão específica pelo número
model = load_model_from_mlflow("PassosMagicos-DefasagemClassifier", version="1")

# Via alias (configure o alias no MLflow UI ou CLI)
model = load_model_from_mlflow("PassosMagicos-DefasagemClassifier", version="champion")
```

### 8.5 Configurar aliases no Model Registry

Aliases permitem marcar versões estáveis para uso em produção. Configure via CLI:

```bash
# Promover versão 1 para alias "champion" (produção)
mlflow models set-alias \
    --model-name PassosMagicos-DefasagemClassifier \
    --version 1 \
    --alias champion

# Consultar informações do modelo
mlflow models get \
    --model-name PassosMagicos-DefasagemClassifier \
    --version 1
```

Ou via Python:

```python
import mlflow

client = mlflow.tracking.MlflowClient()

# Definir alias
client.set_registered_model_alias(
    name="PassosMagicos-DefasagemClassifier",
    alias="champion",
    version="1",
)

# Carregar pelo alias
model_uri = "models:/PassosMagicos-DefasagemClassifier@champion"
model = mlflow.sklearn.load_model(model_uri)
```

### 8.6 Usar servidor MLflow remoto

Para ambientes de time ou CI/CD, configure um servidor MLflow centralizado:

```bash
# Iniciar servidor MLflow com banco SQLite e armazenamento local
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 0.0.0.0 \
    --port 5000

# Apontar o projeto para o servidor remoto (antes de rodar train.py)
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/train.py
```

> O backend padrão do projeto é `sqlite:///mlruns/mlflow.db`, configurável via variável de ambiente `MLFLOW_TRACKING_URI`.

### 8.7 Estrutura gerada pelo MLflow

```
mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── artifacts/
        │   ├── joblib/model.pkl         ← cópia do model.pkl joblib
        │   ├── model/                   ← modelo no formato MLflow
        │   │   ├── MLmodel
        │   │   ├── model.pkl
        │   │   └── requirements.txt
        │   └── reports/metrics.json
        ├── metrics/
        │   ├── f1_score
        │   ├── roc_auc
        │   └── ...
        ├── params/
        │   ├── clf__n_estimators
        │   └── ...
        └── tags/
            ├── dataset
            └── ...
```

### 8.8 Servir modelo diretamente via MLflow (alternativa ao uvicorn)

```bash
# Servir a versão mais recente do modelo registrado
mlflow models serve \
    --model-uri "models:/PassosMagicos-DefasagemClassifier/latest" \
    --host 0.0.0.0 \
    --port 8080

# Testar
curl -X POST http://localhost:8080/invocations \
    -H "Content-Type: application/json" \
    -d '{"dataframe_records": [{"inde": 6.2, "ida": 7.1, "ieg": 5.8, "ipv": 6.5, "iaa": 7.0, "nota_mat": 6.0, "nota_port": 6.5, "pedra_num": 1.0, "avg_notas": 6.25, "eng_aprendizagem": 4.118}]}'
```

> **Nota:** `mlflow models serve` serve o modelo sklearn diretamente e espera as features já processadas (pós feature engineering). Para o endpoint com pré-processamento completo, prefira a API FastAPI em `app/`.

---

## 9. Executando a API



> **Lembre-se:** ative o venv antes de iniciar a API: `source .venv/bin/activate`

### 9.1 Modo desenvolvimento (com reload automático)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 9.2 Modo produção

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 9.3 Verificar que a API subiu

```bash
curl http://localhost:8000/
```

Resposta esperada:
```json
{
  "message": "Passos Mágicos — API de Predição de Defasagem Escolar",
  "version": "1.0.0"
}
```

### 9.4 Documentação interativa (Swagger)

Acesse no navegador: **http://localhost:8000/docs**

Lá você pode explorar todos os endpoints, ver os schemas de entrada/saída e fazer chamadas de teste diretamente pela interface.

Documentação alternativa (ReDoc): **http://localhost:8000/redoc**

---

## 10. Usando a API — Endpoints

### `GET /`

Confirmação de que a API está no ar.

**Resposta:**
```json
{
  "message": "Passos Mágicos — API de Predição de Defasagem Escolar",
  "version": "1.0.0"
}
```

---

### `GET /health`

Verifica se o modelo está carregado e a API está operacional.

**Resposta (modelo OK):**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**Resposta (modelo não carregado):**
```json
{
  "status": "degraded",
  "model_loaded": false
}
```

---

### `POST /predict`

Classifica um aluno como defasado ou no nível com base nos indicadores de desempenho.

#### Schema de entrada (`StudentFeatures`)

| Campo | Tipo | Obrigatório | Intervalo | Descrição |
|-------|------|-------------|-----------|-----------|
| `inde` | float | Sim | 0.0 – 10.0 | INDE — Índice de Desenvolvimento Educacional (2022) |
| `ida` | float | Sim | 0.0 – 10.0 | IDA — Indicador de Desempenho Acadêmico |
| `ieg` | float | Sim | 0.0 – 10.0 | IEG — Indicador de Engajamento |
| `ipv` | float | Sim | 0.0 – 10.0 | IPV — Indicador de Ponto de Virada |
| `iaa` | float | Sim | 0.0 – 10.0 | IAA — Indicador de Auto-Avaliação |
| `nota_mat` | float | Não | 0.0 – 10.0 | Nota de Matemática (imputada com mediana se ausente) |
| `nota_port` | float | Não | 0.0 – 10.0 | Nota de Português (imputada com mediana se ausente) |

#### Schema de saída (`PredictionResponse`)

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `prediction` | int | `0` = no nível, `1` = defasado |
| `label` | string | `"no nível"` ou `"defasado"` |
| `probability_defasado` | float | Probabilidade de defasagem (classe 1) |
| `probability_no_nivel` | float | Probabilidade de estar no nível (classe 0) |
| `model_version` | string | Versão do modelo em uso |

#### Erros possíveis

| Código HTTP | Situação |
|-------------|----------|
| `422 Unprocessable Entity` | Campo obrigatório ausente ou valor fora do intervalo permitido |
| `503 Service Unavailable` | Modelo não carregado (reinicie a API) |

---

## 11. Exemplos de Requisição

### Aluno com bom perfil de indicadores

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inde": 7.5,
    "ida": 8.0,
    "ieg": 7.0,
    "ipv": 8.5,
    "iaa": 7.5,
    "nota_mat": 8.0,
    "nota_port": 7.5
  }'
```

### Aluno com indicadores de risco

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inde": 4.2,
    "ida": 3.5,
    "ieg": 4.0,
    "ipv": 3.8,
    "iaa": 4.5,
    "nota_mat": 4.0,
    "nota_port": 3.5
  }'
```

### Sem informar as notas (campos opcionais)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inde": 5.8,
    "ida": 6.2,
    "ieg": 5.5,
    "ipv": 6.0,
    "iaa": 6.5
  }'
```

> Quando `nota_mat` ou `nota_port` são omitidos, o modelo usa a mediana do conjunto de treino para imputação — comportamento idêntico ao treinamento.

### Exemplo de resposta

```json
{
  "prediction": 1,
  "label": "defasado",
  "probability_defasado": 0.8312,
  "probability_no_nivel": 0.1688,
  "model_version": "1.0.0"
}
```

### Usando Python (requests)

```python
import requests

url = "http://localhost:8000/predict"

payload = {
    "inde": 6.2,
    "ida": 7.1,
    "ieg": 5.8,
    "ipv": 6.5,
    "iaa": 7.0,
    "nota_mat": 6.0,
    "nota_port": 6.5,
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Predição : {result['label']}")
print(f"Probabilidade de defasagem: {result['probability_defasado']:.1%}")
```

### Classificação em lote (Python)

```python
import requests
import pandas as pd

url = "http://localhost:8000/predict"

alunos = [
    {"inde": 7.5, "ida": 8.0, "ieg": 7.0, "ipv": 8.5, "iaa": 7.5},
    {"inde": 4.2, "ida": 3.5, "ieg": 4.0, "ipv": 3.8, "iaa": 4.5},
    {"inde": 5.9, "ida": 6.1, "ieg": 5.8, "ipv": 6.0, "iaa": 6.2},
]

resultados = []
for aluno in alunos:
    resp = requests.post(url, json=aluno).json()
    resultados.append({
        "inde": aluno["inde"],
        "predição": resp["label"],
        "prob_defasado": f"{resp['probability_defasado']:.1%}",
    })

df = pd.DataFrame(resultados)
print(df.to_string(index=False))
```

---

## 12. Testes

### 12.1 Executar todos os testes unitários

```bash
pytest tests/ -v
```

### 12.2 Com relatório de cobertura

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 12.3 Apenas testes rápidos (sem integração com dataset)

```bash
pytest tests/ -v -m "not integration"
```

### 12.4 Apenas testes de integração (usa o dataset real)

```bash
pytest tests/ -v -m "integration"
```

### 12.5 Resultado esperado

```
101 passed in ~2s

Name                         Stmts   Miss  Cover
------------------------------------------------
src/__init__.py                  0      0   100%
src/evaluate.py                 36      0   100%
src/feature_engineering.py      39      0   100%
src/preprocessing.py            70      5    93%
src/train.py                    56     10    82%
src/utils.py                    19      0   100%
------------------------------------------------
TOTAL                          220     28    87%
```

**Cobertura: 87%** — acima do mínimo exigido de 80%.

### 12.6 Organização dos testes

| Arquivo | Módulo testado | Nº de testes |
|---------|---------------|-------------|
| `test_preprocessing.py` | `src/preprocessing.py` | 23 |
| `test_feature_engineering.py` | `src/feature_engineering.py` | 21 |
| `test_evaluate.py` | `src/evaluate.py` | 14 |
| `test_train.py` | `src/train.py` | 10 (+6 integração) |
| `test_utils.py` | `src/utils.py` | 11 |
| `test_mlflow_integration.py` | MLflow tracking/registry | 10 (+2 integração) |

---

## 13. Detalhes do Modelo

### 12.1 Features utilizadas

| Feature | Origem | Descrição |
|---------|--------|-----------|
| `inde` | Dataset (INDE 22) | Índice de Desenvolvimento Educacional |
| `ida` | Dataset (IDA) | Indicador de Desempenho Acadêmico |
| `ieg` | Dataset (IEG) | Indicador de Engajamento |
| `ipv` | Dataset (IPV) | Indicador de Ponto de Virada |
| `iaa` | Dataset (IAA) | Indicador de Auto-Avaliação |
| `nota_mat` | Dataset (Matem) | Nota de Matemática |
| `nota_port` | Dataset (Portug) | Nota de Português |
| `pedra_num` | Derivada | Classificação PEDRA do INDE (0=Quartzo … 3=Topázio) |
| `avg_notas` | Derivada | Média aritmética entre nota_mat e nota_port |
| `eng_aprendizagem` | Derivada | IEG × IDA / 10 — engajamento ponderado pelo desempenho |

### 12.2 Classificação PEDRA (feature `pedra_num`)

| INDE | PEDRA | Valor numérico |
|------|-------|---------------|
| 0.000 – 5.506 | Quartzo | 0 |
| 5.506 – 6.868 | Ágata | 1 |
| 6.868 – 8.230 | Ametista | 2 |
| 8.230 – 10.000 | Topázio | 3 |

### 12.3 Pipeline de pré-processamento (inferência)

```
Entrada (JSON) → preprocess_inference() → engineer_features()
    │
    ▼
SimpleImputer (mediana do treino)   ← trata nota_mat/nota_port ausentes
    │
    ▼
StandardScaler                      ← normaliza para média 0, desvio 1
    │
    ▼
GradientBoostingClassifier          ← predição + probabilidades
```

### 12.4 Métricas no conjunto de teste

| Métrica | Valor |
|---------|-------|
| **F1-Score** | **0.9249** |
| **AUC-ROC** | **0.9245** |
| Precisão | 0.8797 |
| Recall | 0.9750 |
| Acurácia | 0.8895 |

**Matriz de Confusão** (172 amostras de teste):

```
                Previsto: no nível   Previsto: defasado
Real: no nível       36 (TN)              16 (FP)
Real: defasado        3 (FN)             117 (TP)
```

**Interpretação:**
- **Recall = 0.975** → apenas 3 alunos defasados deixaram de ser identificados
- **FP = 16** → 16 alunos no nível foram sinalizados como defasados (falso alarme aceitável)
- Para o domínio de educação, priorizar recall alto é correto: é melhor investigar um caso a mais do que deixar um aluno em risco sem suporte

### 12.5 Melhores hiperparâmetros encontrados

```
learning_rate  = 0.1
max_depth      = 3
n_estimators   = 200
subsample      = 1.0
```

### 12.6 Dados de treinamento

| Característica | Valor |
|---------------|-------|
| Dataset | BASE DE DADOS PEDE 2024 - DATATHON.xlsx |
| Total de amostras | 860 |
| Treino (80%) | 688 amostras |
| Teste (20%) | 172 amostras |
| Split | Estratificado (mantém proporção das classes) |
| Alunos defasados | 601 (69.9%) |
| Alunos no nível | 259 (30.1%) |

---

## 14. Decisões Técnicas Relevantes

### Por que IAN foi excluído das features?

O **IAN (Indicador de Adequação de Nível)** foi removido das features do modelo por ser uma recodificação direta da coluna `Defas` (o target):

| IAN | Defas correspondente |
|-----|---------------------|
| 10.0 | 0, +1, +2 (no nível ou adiantado) |
| 5.0 | -1, -2 (defasado por 1–2 anos) |
| 2.5 | -3, -4, -5 (defasado por 3+ anos) |

Usar IAN como feature causava **data leakage**: o modelo atingia F1=1.00 artificialmente porque recebia o target recodificado como entrada. A correlação de Pearson confirmou: `IAN ↔ target = -0.98`.

### Por que GradientBoosting?

- Lida bem com relações não-lineares entre os indicadores
- Robusto a pequenas variações de escala (complementado pelo scaler)
- Suporta `n_iter_no_change` para early stopping, evitando overfitting
- Alternativa considerada: RandomForestClassifier (desempenho similar, mas GBM teve F1 CV 0.929 vs 0.917)

### Por que SimpleImputer no pipeline sklearn?

As notas de Matemática e Português são opcionais na API. Em vez de hardcodar as medianas do treino no código de inferência, o `SimpleImputer(strategy="median")` aprende as medianas durante o fit e as aplica automaticamente em `predict`. Isso garante consistência e evita vazamento de informação de treino para inferência fora do pipeline.

### Por que F1-Score como métrica principal?

Com 70% de defasados (desbalanceamento moderado), a acurácia simples seria enganosa — um modelo que classifica tudo como defasado teria 70% de acurácia. O F1-Score equilibra precisão e recall, sendo mais informativo para dados desbalanceados.

---

## 15. Monitoramento e Logs

### 15.1 Métricas em tempo real — Prometheus + Grafana

A API expõe o endpoint `/metrics` (formato Prometheus) via `prometheus-fastapi-instrumentator`. As métricas coletadas incluem:

| Métrica | O que mede |
|---------|-----------|
| `http_requests_total` | Total de requisições por rota e status code |
| `http_request_duration_seconds` | Latência (histograma) por rota |
| `http_requests_in_progress` | Requisições em andamento no momento |

Para visualizar graficamente, suba a stack completa:

```bash
docker compose up --build
```

Acesse o Grafana em http://localhost:3000 e importe o dashboard ID `19797`.

### 15.2 Arquivos de log

| Arquivo | Gerado por | Conteúdo |
|---------|-----------|---------|
| `logs/train.log` | `python src/train.py` | Métricas do treino, hiperparâmetros, tempo |
| `logs/api.log` | Aplicação FastAPI | Cada predição com label, probabilidade e latência |

### 15.3 Formato dos logs

```
2024-11-15 10:23:45 | INFO     | app.routes | Predição: defasado | prob_defasado=0.832 | latência=12.3ms
2024-11-15 10:23:51 | INFO     | app.routes | Predição: no nível | prob_defasado=0.218 | latência=9.8ms
```

### 15.4 Acompanhar logs em tempo real

```bash
# Logs da API
tail -f logs/api.log

# Logs do treino
tail -f logs/train.log
```

### 15.5 Verificar métricas salvas

```bash
cat models/metrics.json
```

---

## 16. CI/CD — Integração e Entrega Contínua

O projeto usa **GitHub Actions** para automação do ciclo de desenvolvimento. Os workflows estão em `.github/workflows/`.

### 16.1 Pipeline de CI (`ci.yml`)

Disparado a cada push em qualquer branch e em pull requests para `main`:

1. Instala dependências
2. Executa os 101 testes (`pytest -m "not integration"`)
3. Faz o build da imagem Docker para validar o `Dockerfile`

### 16.2 Pipeline de CD (`cd.yml`)

Disparado a cada merge na `main`:

1. Faz o build da imagem Docker
2. Publica no Docker Hub com duas tags: `latest` e o SHA do commit (para rollback)

### 16.3 Configurar secrets no GitHub

No repositório: **Settings > Secrets and variables > Actions > New repository secret**

| Secret | Valor |
|--------|-------|
| `DOCKERHUB_USERNAME` | Seu usuário do Docker Hub |
| `DOCKERHUB_TOKEN` | Token em Docker Hub > Account Settings > Security > New Access Token |

### 16.4 Forçar execução manual

```bash
# Push vazio para disparar o CI
git commit --allow-empty -m "ci: force run"
git push
```

Ou pelo GitHub: **Actions > selecione o workflow > Run workflow**.

Para mais detalhes sobre a estratégia de CI/CD, consulte o arquivo `CICD_OVERVIEW.md`.

---

## 17. Resolução de Problemas

### `ModuleNotFoundError: No module named 'src'`

Execute sempre a partir da **raiz do projeto**, não de dentro de `src/`, e com o venv ativado:

```bash
# Correto
cd passos_magicos_datathon
source .venv/bin/activate
python src/train.py

# Incorreto
cd src
python train.py
```

### `FileNotFoundError: Modelo não encontrado: models/model.pkl`

O modelo precisa ser treinado antes de iniciar a API (com o venv ativado):

```bash
python src/train.py
uvicorn app.main:app --reload
```

### `[Errno 98] Address already in use` (porta 8000 ocupada)

```bash
# Verificar o processo usando a porta
lsof -i :8000

# Usar porta alternativa
uvicorn app.main:app --port 8001
```

### Erro `422 Unprocessable Entity` ao chamar `/predict`

Verifique se:
- Todos os campos obrigatórios estão presentes: `inde`, `ida`, `ieg`, `ipv`, `iaa`
- Os valores estão no intervalo `[0.0, 10.0]`
- O `Content-Type: application/json` está no cabeçalho

### Cobertura de testes abaixo do esperado

Para incluir os testes de integração (que cobrem `train.py`):

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### MLflow: `mlflow` não reconhecido como comando

Certifique-se de que o ambiente virtual está ativado. Com o `.venv` ativo, `mlflow` fica disponível automaticamente em `.venv/bin/mlflow`:

```bash
# Ativar o venv (se ainda não estiver ativo)
source .venv/bin/activate

# Verificar
mlflow --version
```

### MLflow: `RESOURCE_DOES_NOT_EXIST` ao carregar modelo do registry

O modelo ainda não foi registrado. Execute `python src/train.py` ao menos uma vez antes de chamar `load_model_from_mlflow()`.

### MLflow: runs duplicados no mesmo experimento

Comportamento esperado — cada execução de `train.py` cria um novo run. Para ver apenas o melhor, filtre pela métrica `f1_score` decrescente na UI (`mlflow ui`).

### MLflow: tracking URI diferente entre treino e UI

Certifique-se de que `--backend-store-uri` do `mlflow ui` aponta para o mesmo diretório usado durante o treino (padrão: `./mlruns` na raiz do projeto).

---

## Autores

Desenvolvido como projeto final do **Datathon FIAP Pós Tech — ML Engineering 2024**.

- Dataset e contexto: [ONG Passos Mágicos](https://passosmagicos.org.br/)
