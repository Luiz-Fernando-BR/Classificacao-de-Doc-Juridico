import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Dicionário de mapeamento para as labels
LABEL_MAP = {
    0: 'Contrato',
    1: 'Petição',
    2: 'Sentença'
}

# Função para extrair texto de um documento PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Função para classificar o tipo de documento jurídico
def classify_document(text, classifier):
    result = classifier(text)
    label = result[0]['label']
    
    # Mapear o label numérico para o nome do documento
    label_number = int(label.split('_')[1])  # Extrair o número do label (LABEL_0, LABEL_1, etc.)
    return LABEL_MAP.get(label_number, "Desconhecido")

# Carregar o classificador da Hugging Face (modelo e tokenizer treinados)
model_name = './modelo_treinado'  # Caminho do modelo treinado
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# Widget do Streamlit para upload do documento PDF
uploaded_file = st.file_uploader("Faça upload de um documento PDF", type="pdf")

# Verifica se um documento foi carregado
if uploaded_file is not None:
    # Extrai o texto do PDF
    text = extract_text_from_pdf(uploaded_file)
    
    # Limpa o texto extraído (opcional)
    cleaned_text = text.lower().replace('\n', ' ').strip()
    
    if st.button('Classificar Documento'):
        # Classifica o tipo de documento
        classification = classify_document(cleaned_text, classifier)
        
        # Exibe a classificação
        st.subheader("Classificação do Documento:")
        st.write(classification)