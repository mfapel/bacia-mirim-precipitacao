# Monitoramento de Precipitação — Bacia Mirim

Website de acompanhamento do acumulado de precipitação nas estações automáticas INMET
da região da Bacia Hidrográfica da Lagoa Mirim (RS/Brasil).

## Funcionalidades

- **Mapa interativo** com todas as estações da região, coloridas por intensidade de chuva
- **Clique em qualquer estação** para ver a série temporal diária
- **Período configurável**: últimas 24h, 3, 7, 15 ou 30 dias
- **Tabela comparativa** e ranking de acumulados
- Dados atualizados automaticamente a cada 30 minutos

## Rodar localmente

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Iniciar o app
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador.

## Deploy gratuito (Streamlit Community Cloud)

1. Faça fork/push deste repositório para o GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Clique em **New app** → selecione o repositório → `app.py`
4. Clique em **Deploy** — URL gerada: `seuprojeto.streamlit.app`

## Estrutura

```
bacia-mirim-precipitacao/
├── app.py              # Interface principal (Streamlit)
├── inmet_api.py        # Acesso à API pública do INMET
├── requirements.txt    # Dependências Python
└── .streamlit/
    └── config.toml     # Tema e configurações
```

## Fonte dos dados

- **API em tempo real:** `https://apitempo.inmet.gov.br` (pública, sem autenticação)
- **Série histórica:** BDMEP `https://bdmep.inmet.gov.br` (requer token — integração futura)
- Cobertura: estações automáticas no bounding box da Bacia Mirim
  (lat -34° a -30.5°, lon -54.5° a -49.5°)

## Limitações atuais

- Dados brutos não validados (estações automáticas)
- Série histórica longa (pré-2024) requer integração com BDMEP + token
- Estações podem ter lacunas de transmissão
