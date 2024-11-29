import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import torch

# Função para extrair texto de um documento PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Função para preparar o dataset com limpeza de texto
def prepare_dataset():
    contrato_text = extract_text_from_pdf(r"C:\caminho\para\os\arquivos\Contrato_Exemplo_SemPrimeiraLinha.pdf")
    peticao_text = extract_text_from_pdf(r"C:\caminho\para\os\arquivos\Peticao_Judicial_Exemplo_SemPrimeiraLinha.pdf")
    sentenca_text = extract_text_from_pdf(r"C:\caminho\para\os\arquivos\Sentenca_Judicial_Exemplo_SemPrimeiraLinha.pdf")


    def clean_text(text):
        text = text.lower()  # Converte para minúsculas
        return text

    contrato_text = clean_text(contrato_text)
    peticao_text = clean_text(peticao_text)
    sentenca_text = clean_text(sentenca_text)

    contrato_examples = contrato_text.split('Contrato_Exemplo_SemPrimeiraLinha.pdf')
    peticao_examples = peticao_text.split('Peticao_Judicial_Exemplo_SemPrimeiraLinha.pdf')
    sentenca_examples = sentenca_text.split('Sentenca_Judicial_Exemplo_SemPrimeiraLinha.pdf')

    data = {
        'text': contrato_examples + peticao_examples + sentenca_examples,
        'label': [0] * len(contrato_examples) + [1] * len(peticao_examples) + [2] * len(sentenca_examples)
    }

    dataset = DatasetDict({
        'train': Dataset.from_dict(data),
        'test': Dataset.from_dict(data)
    })
    return dataset

# Função para treinar o modelo
def train_model(dataset, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        logging_dir='./logs',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
    )

    trainer.train()
    trainer.save_model('./modelo_treinado')  # Salvar o modelo treinado
    tokenizer.save_pretrained('./modelo_treinado')  # Salvar o tokenizer
    return trainer

# Preparar dataset e treinar o modelo
model_name = "neuralmind/bert-base-portuguese-cased"
dataset = prepare_dataset()
train_model(dataset, model_name)