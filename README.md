# Classificação de Documentos Jurídicos com Modelos de Machine Learning

Este projeto utiliza modelos de aprendizado de máquina para classificar documentos jurídicos em três categorias: **Contrato**, **Petição** e **Sentença**. O modelo é baseado em **BERT**, uma arquitetura Transformer, e foi treinado com textos jurídicos extraídos de documentos em formato PDF. O classificador foi desenvolvido usando a biblioteca **Hugging Face Transformers**.

## Funcionalidades

- **Extração de Texto de PDFs**: Utiliza a biblioteca **PyMuPDF (fitz)** para extrair texto de documentos PDF.
- **Classificação de Documentos Jurídicos**: Classifica os documentos em três categorias específicas: **Contrato**, **Petição** e **Sentença**.
- **Interface de Usuário com Streamlit**: Através de um aplicativo web simples com **Streamlit**, permite o upload de documentos PDF para classificação em tempo real.

## Como Usar

### Pré-requisitos

Antes de rodar o arquivo `Docanalise.py`, é necessário treinar o modelo primeiro com o arquivo `treinamentodocjuridico.py`. O treinamento do modelo utiliza um conjunto de dados de exemplo em formato PDF. O modelo treinado é salvo e poderá ser carregado posteriormente pelo `Docanalise.py` para realizar a classificação.

### Passos para Rodar o Projeto

1. **Treinar o Modelo (Primeiro Passo)**:
   - Execute o arquivo `treinamentodocjuridico.py` para treinar o modelo com os documentos de exemplo. Isso gerará o modelo treinado e o tokenizer, que serão salvos no diretório `modelo_treinado`.
   
   **Comando para executar:**
   ```bash
   python treinamentodocjuridico.py
   ```

2. **Classificar Documentos (Segundo Passo)**:
   - Após treinar o modelo, execute o arquivo `Docanalise.py` para classificar novos documentos jurídicos. O arquivo `Docanalise.py` permite o upload de documentos PDF através de uma interface do Streamlit.
   
   **Comando para executar:**
   ```bash
   streamlit run docanalise1.py
   ```

### Bibliotecas Necessárias

Para garantir que o projeto funcione corretamente, instale as bibliotecas necessárias listadas abaixo:

1. **Transformers**: Para carregar e usar os modelos da Hugging Face.
2. **PyMuPDF (fitz)**: Para extrair texto de arquivos PDF.
3. **Torch**: Biblioteca de deep learning necessária para executar modelos pré-treinados.
4. **Streamlit**: Para criar a interface de usuário para upload e classificação de documentos.

### Instalação das Dependências

Execute o comando abaixo para instalar todas as dependências:

```bash
pip install transformers torch pymupdf streamlit
```

### Estrutura de Arquivos

- **`treinamentodocjuridico.py`**: Script responsável por treinar o modelo e salvar o modelo treinado.
- **`Docanalise.py`**: Script para carregar o modelo treinado e realizar a classificação de documentos PDF.
- **`./modelo_treinado`**: Diretório onde o modelo e o tokenizer treinados são salvos.

## Como Funciona

1. **Treinamento**:
   - O arquivo `treinamentodocjuridico.py` faz a extração e pré-processamento de documentos PDF, como contratos, petições e sentenças.
   - Ele utiliza o modelo pré-treinado **BERT** da Hugging Face, realizando o treinamento sobre um conjunto de documentos jurídicos, e salva o modelo treinado em um diretório `modelo_treinado`.

2. **Classificação**:
   - O arquivo `Docanalise.py` permite o upload de documentos em formato PDF. Após o upload, o texto é extraído e classificado com o modelo treinado.
   - O modelo classifica os documentos em três categorias: **Contrato**, **Petição** e **Sentença**.
   - O resultado da classificação é exibido na interface do **Streamlit**.

## Observações

- **Dependências**: Este projeto foi desenvolvido e testado em Python 3.12.4 Certifique-se de ter o Python e as dependências corretamente instaladas.
- **Ajustes no Modelo**: Caso queira treinar com outros documentos ou modificar a arquitetura do modelo, ajuste o código no arquivo `treinamentodocjuridico.py`.
- **Resultados**: Os documentos classificados terão como resultado a label correspondente a uma das três categorias, que será exibida como "Contrato", "Petição" ou "Sentença".
