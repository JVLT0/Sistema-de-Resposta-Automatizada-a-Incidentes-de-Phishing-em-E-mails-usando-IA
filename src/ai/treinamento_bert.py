import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import pandas as pd
import os

# Configura caminhos base considerando que o script está em src/ai/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Volta para a raiz do projeto
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset personalizado
class EmailDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        texto = row['assunto'] + " " + row['conteudo']
        label = row['is_phishing']
        
        encoding = self.tokenizer(texto, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

class TreinadorBERT:
    def __init__(self, model, tokenizer, dataset, batch_size=8, epochs=3, lr=2e-5):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def treinar(self):
        logger.info("Iniciando treinamento do BERT...")
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            logger.info(f"Época {epoch + 1}, Loss: {avg_loss:.4f}")
            self.avaliar()
        
        # Salva o modelo na pasta models/
        os.makedirs(MODELS_DIR, exist_ok=True)
        modelo_path = os.path.join(MODELS_DIR, "modelo_bert_phishing.pth")
        torch.save(self.model.state_dict(), modelo_path)
        logger.info(f"Modelo salvo em: {modelo_path}")

    def avaliar(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        logger.info(f"Acurácia de validação: {100 * correct / total:.2f}%")

def main():
    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=2)
    
    # Carrega dados da pasta data/
    dataset_path = os.path.join(DATA_DIR, "dataset_emails.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Arquivo de dataset não encontrado em: {dataset_path}")
    
    dataset_emails = pd.read_csv(dataset_path)
    logger.info(f"Dataset carregado: {len(dataset_emails)} registros")

    dataset = EmailDataset(dataset_emails, tokenizer, max_length=256)
    
    treinador = TreinadorBERT(model, tokenizer, dataset)
    treinador.treinar()

if __name__ == "__main__":
    main()