# CI/CD — Visão Geral Aplicada ao Projeto Passos Mágicos

## O que é CI/CD?

**CI (Integração Contínua)** e **CD (Entrega/Deploy Contínuo)** são práticas de engenharia de software que automatizam o caminho entre uma alteração de código e sua disponibilização em produção. O objetivo é reduzir erros manuais, detectar problemas cedo e tornar o processo de entrega previsível e repetível.

---

## Como se aplica a este projeto

Este projeto tem três dimensões que o CI/CD precisa cobrir:

1. **Código** — módulos Python de pré-processamento, feature engineering e treinamento
2. **Modelo de ML** — artefato treinado com métricas rastreáveis
3. **API de predição** — serviço FastAPI containerizado via Docker

---

## Pipeline de CI (Integração Contínua)

A cada push ou pull request, o pipeline executa automaticamente:

### 1. Testes automatizados
```
pytest tests/ --cov=src
```
O projeto possui **101 testes** cobrindo 87% do código-fonte, validando:
- Pré-processamento e feature engineering
- Treinamento do modelo e persistência
- Comportamento dos endpoints `/health` e `/predict`
- Integração com MLflow

**Por que isso importa:** garante que uma mudança no código de pré-processamento não quebre silenciosamente a predição em produção.

### 2. Qualidade de código
Ferramentas como `flake8` ou `ruff` verificam padrões de estilo e erros estáticos antes do merge.

### 3. Build do container
```
docker build -t passos-magicos .
```
O `Dockerfile` já está otimizado com camadas cacheáveis (dependências separadas do código) e inclui um `HEALTHCHECK` nativo. O build no CI confirma que a imagem compila corretamente antes de qualquer deploy.

---

## Pipeline de CD (Entrega Contínua)

Após o CI passar, o CD automatiza a entrega:

### 1. Publicação da imagem
A imagem Docker é publicada em um registry (ex: Docker Hub, AWS ECR, GitHub Container Registry) com a tag da versão ou do commit.

### 2. Deploy do modelo via MLflow
O projeto usa **MLflow** como Model Registry. O fluxo de promoção de modelo é:

```
Treino local → Experimento registrado → Staging → Production
```

A API carrega o modelo diretamente do registry pelo nome e versão, sem necessidade de copiar arquivos manualmente. No pipeline CD, após um retreino aprovado, a versão é promovida via script:

```python
client.transition_model_version_stage(
    name="PassosMagicos-DefasagemClassifier",
    version="3",
    stage="Production"
)
```

### 3. Deploy da API
O container atualizado é implantado no ambiente alvo (ex: VM, Kubernetes, AWS ECS). O `HEALTHCHECK` do Docker garante que instâncias não saudáveis sejam substituídas automaticamente.

---

## Ferramenta recomendada: GitHub Actions

**Por que GitHub Actions:**
- Nativo ao repositório, sem infraestrutura adicional para o CI/CD em si
- Gratuito para repositórios públicos; custos baixos para privados
- YAML simples, fácil de auditar e versionar junto com o código
- Integra nativamente com Docker Hub, AWS, Azure e GCP

Estrutura mínima de workflow:

```
.github/
  workflows/
    ci.yml    ← roda testes e build a cada push
    cd.yml    ← publica imagem e promove modelo após merge na main
```

---

## Resumo do Fluxo

```
Desenvolvedor faz push
        |
        v
[CI] Testes + Lint + Docker build
        |
    (aprovado)
        |
        v
[CD] Publica imagem no registry
        |
        v
[CD] Promove modelo no MLflow (Staging → Production)
        |
        v
[CD] Deploy da API com novo container
        |
        v
Healthcheck confirma que a API está respondendo
```

---

## Benefícios concretos para o projeto

| Sem CI/CD | Com CI/CD |
|---|---|
| Deploy manual, propenso a erro | Deploy automatizado e auditável |
| Modelo e API podem estar dessincronizados | Model Registry garante rastreabilidade da versão em produção |
| Bug detectado em produção | Bug detectado nos 101 testes antes do merge |
| Rollback manual e demorado | Revert de imagem ou versão de modelo em minutos |

---

## Consideracoes Finais

O ponto mais relevante para projetos de ML é que CI/CD vai além do codigo: ele precisa cobrir o **ciclo de vida do modelo**. O MLflow, ja integrado a este projeto, e a peca que preenche essa lacuna — conectando o experimento de treino ao modelo em producao de forma rastreavel e controlada.
