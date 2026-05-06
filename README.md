# Radar GARCH B3 — Travas de Put com Decisão Pronta

App em Streamlit para acessar pelo celular ou tablet Android. A estratégia implementada é:

- vender put na banda inferior de -1,5 desvio GARCH;
- comprar put na banda inferior de -2 desvios GARCH;
- priorizar blue chips/empresas fortes que podem virar carteira;
- limitar a seleção a no máximo 2 bancos;
- filtrar por preço, liquidez, tendência, ATR, suporte, IV Rank e crédito da trava;
- calcular quantidade sugerida, ganho máximo, perda máxima e preço efetivo se houver exercício;
- considerar rolagem de até 3 vezes e, na 4ª ameaça, decidir se aceita exercício.

## Arquivos

- `app.py`: arquivo principal para publicar no Streamlit Cloud.
- `app_radar_garch_puts_b3.py`: cópia com nome descritivo.
- `requirements.txt`: dependências para o Streamlit Cloud.

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Como publicar

1. Crie um repositório no GitHub.
2. Envie `app.py` e `requirements.txt`.
3. Entre no Streamlit Community Cloud.
4. Clique em New app.
5. Escolha o repositório.
6. Main file path: `app.py`.
7. Clique em Deploy.

## Uso semanal

- Sexta após fechamento: gerar decisão pronta para a próxima semana.
- Segunda depois de 10h30: rodar novamente, confirmar prêmios reais no home broker e executar manualmente.

## Dados

- Preços/volume dos ativos: Yahoo Finance via yfinance.
- Volatilidade: GARCH(1,1), com fallback EWMA.
- Opções: tentativa de leitura pública do Opções.net.br e/ou CSV manual.

## Observação importante

O app não envia ordens, não garante lucro e não substitui conferência no home broker.
