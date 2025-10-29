# Fraude Cartao de Credito

Pipeline completo para detectar transacoes fraudulentas em um dataset de cartao de credito. O codigo transforma o notebook original em um script Python capaz de rodar localmente, executando preprocessamento, selecao de atributos, tecnicas de balanceamento, avaliacao de diversos modelos e consolidacao dos resultados.

Link para testar: https://colab.research.google.com/drive/1u8qnkADP7HDaiWatfLm9xBeLZyG93ltv?usp=sharing

## Conteudo principal

- Carga do dataset a partir de um arquivo CSV informado por argumento ou via prompt.
- Exploracao inicial: estatisticas, distribuicao das classes e visualizacoes.
- Normalizacao, selecao de atributos e criacao de subconjuntos balanceados (SMOTE, undersampling e SMOTETomek).
- Treino, busca de hiperparametros e ensembles com modelos da biblioteca scikit-learn e XGBoost.
- Analises finais: matriz de confusao, relatorio de classificacao, curva ROC e resumo textual dos principais achados.

## Requisitos

- Python 3.10 ou superior.
- pip configurado.
- Bibliotecas:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - imbalanced-learn
  - xgboost

Instale as dependencias com:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

## Como executar

1. Posicione o arquivo CSV (exemplo: `creditcard.csv`) na pasta que preferir.
2. Execute o script apontando para o caminho do dataset:

```bash
python fraudecartaocredito.py --dataset "caminho/para/creditcard.csv"
```

Sem o argumento `--dataset`, o script solicita o caminho via prompt interativo.

Os resultados sao exibidos diretamente no terminal e graficos surgem em janelas interativas.

## Estrutura de saida

Durante a execucao o script apresenta:

- Estatisticas descritivas do dataset.
- Distribuicao das classes (legitima vs fraude) e graficos.
- Desempenho de cada combinacao (modelo + tecnica de balanceamento) exibido em tabelas ordenadas por macro F1-score.
- Matriz de confusao, curva ROC, relatorio de classificacao e resumo textual do melhor modelo.

Nenhum arquivo adicional e gerado por padrao; os resultados sao impressos e exibidos na tela.

## Dataset sugerido

Este projeto foi pensado para o dataset publico de fraude em cartoes de credito disponibilizado no Kaggle (MLG-ULB). Qualquer arquivo CSV com coluna alvo binaria pode ser usado, desde que o nome da coluna alvo seja `Class` ou esteja na ultima coluna do arquivo.

## Observacoes

- Todos os comentarios e mensagens foram convertidos para ASCII para evitar problemas de codificacao.
- Ajuste hiperparametros, tecnicas de balanceamento ou estrutura do ensemble conforme a necessidade do seu problema real.
- Caso deseje registrar os resultados em arquivos, adapte as secoes finais do script para salvar tabelas e graficos.

