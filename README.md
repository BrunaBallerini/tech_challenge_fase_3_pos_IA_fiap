# Tech Challenge 3 — Fine-Tuning de Modelo LLaMA 3 8B (4-bit) via Unsloth

## Descrição do Projeto

Este projeto implementa um fluxo completo de fine-tuning para um modelo de linguagem de grande porte (LLM) utilizando o **LLaMA 3 8B (4-bit)** com a biblioteca **Unsloth**.
O objetivo é aprimorar a capacidade do modelo em **gerar descrições de produtos** a partir de títulos, simulando perguntas de usuários e avaliando o desempenho antes e depois do ajuste fino.

## Objetivo

A tarefa principal consiste em:

1. Receber uma pergunta do usuário sobre um **título de produto**.
2. Recuperar o **título mais similar** usando busca fuzzy.
3. Gerar uma **descrição aprendida** a partir do modelo ajustado.
4. Comparar o resultado antes e depois do fine-tuning.

## Etapas do Projeto

O notebook executa o seguinte pipeline:

1. **Carregamento e limpeza do dataset** (`trn.json`) — remoção de registros com conteúdo vazio.
2. **Geração de arquivo limpo:** `data_titles_contents_cleaned.jsonl`.
3. **Conversão para formato de instrução:** `formatted_products_chat_data.json` (campos: instruction / input / output).
4. **Preparação dos prompts** no formato Alpaca-like.
5. **Divisão** entre dados de treino e validação.
6. **Baseline** de inferência antes do fine-tuning.
7. **Fine-tuning** LoRA 4-bit utilizando o modelo `unsloth/llama-3-8b-bnb-4bit`.
8. **Avaliação** pós-treino com geração de respostas.
9. **Cálculo de métricas** (ROUGE-1/2/L, BLEU, overlap de tokens).
10. **Salvamento** dos adaptadores LoRA e modelo mesclado.
11. **Função de busca fuzzy** de títulos e geração de descrições.
12. **Pipeline interativo** com logs e exportação de resultados.

## Tecnologias Utilizadas

* Python 3.10+
* [Unsloth](https://github.com/unslothai/unsloth)
* Hugging Face Transformers
* Datasets / Accelerate / TRL / PEFT / BitsAndBytes
* RapidFuzz (busca de similaridade textual)
* Evaluate / SacreBLEU (métricas)
* Google Colab (ambiente de execução)

## Como Executar

1. **Clone o repositório** ou carregue o notebook no Google Colab.
2. **Monte o Google Drive** (ou ajuste caminhos locais).
3. **Instale as dependências:**

   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   pip install transformers datasets accelerate peft bitsandbytes trl rapidfuzz evaluate sacrebleu
   ```
4. **Execute as células em sequência**, garantindo acesso à GPU de alta memória (20GB+ RAM).
5. O notebook gerará:

   * Arquivos de dados limpos e formatados
   * Modelos fine-tunados
   * Métricas salvas em `results/metrics.json`

## Avaliação e Resultados

As métricas calculadas incluem:

* **ROUGE-1**, **ROUGE-2**, **ROUGE-L**
* **SacreBLEU**
* Comparação de **similaridade de tokens** entre respostas geradas e referências

O notebook exibe resultados consolidados após o fine-tuning, permitindo análise quantitativa e qualitativa da melhoria.
