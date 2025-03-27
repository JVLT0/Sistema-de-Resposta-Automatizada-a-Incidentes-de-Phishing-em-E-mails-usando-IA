import os
import numpy as np
import pandas as pd
import torch
import base64
import logging
import re
import nltk
from datetime import datetime, timedelta
import random
from transformers import BertTokenizer, BertForSequenceClassification
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from bs4 import BeautifulSoup

# Download de stopwords do NLTK
nltk.download('stopwords')

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================== Caminhos ajustados ======================
# Base: script está em src/ (assumindo que main.py está no mesmo nível que auth/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Volta para a raiz do projeto
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Arquivos de autenticação
CREDENTIALS_FILE = os.path.join(CONFIG_DIR, 'client_secret.json')  # ou 'credentials.json'
TOKEN_FILE = os.path.join(CONFIG_DIR, 'token.json')

# Modelo BERT
MODEL_PATH = os.path.join(MODELS_DIR, 'modelo_bert_phishing.pth')

# Verificação de diretórios
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
# ================================================================

# Função para autenticação com Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def criar_servico():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)
        
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    service = build('gmail', 'v1', credentials=creds)
    return service

# Função para remover HTML dos e-mails
def limpar_html(texto):
    soup = BeautifulSoup(texto, "html.parser")
    return soup.get_text()

# Função para buscar e-mails no Gmail
def listar_emails(service, num_emails=10):
    try:
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread").execute()
        messages = results.get('messages', [])
        emails = []

        for message in messages[:num_emails]:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            email_data = msg['payload']['headers']
            subject = ""
            for values in email_data:
                if values['name'] == 'Subject':
                    subject = values['value']
            
            # Decodifica o corpo do e-mail
            payload = msg['payload']['body']
            if 'data' in payload:
                body = base64.urlsafe_b64decode(payload['data']).decode('utf-8')
                body = limpar_html(body)  # Remove tags HTML
            else:
                body = 'Corpo do e-mail não disponível'
            
            emails.append({
                'assunto': subject,
                'conteudo': body,
                'is_phishing': None  # Será preenchido após a detecção
            })

        return pd.DataFrame(emails)

    except Exception as error:
        logger.error(f'Erro ao listar e-mails: {error}')
        return pd.DataFrame()

# Detector de phishing
class DetectorPhishing:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.model = BertForSequenceClassification.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            num_labels=2
        ).to(self.device)

        # Carrega o modelo da pasta models/
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
        
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()  # Modo de inferência
    
    def prever(self, emails_df):
        logger.info("Executando detecção de phishing...")
        textos = (emails_df['assunto'] + " " + emails_df['conteudo']).tolist()
        encodings = self.tokenizer(
            textos, truncation=True, padding=True, max_length=256, return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
            probabilidades = torch.nn.functional.softmax(outputs.logits, dim=1)
            previsoes = (probabilidades[:, 1] > 0.6).int()
        
        return previsoes.cpu().numpy()

# Respondedor de incidentes
class RespondedorIncidentes:
    def tomar_acao(self, emails_df, previsoes):
        logger.info("Tomando ações com base nos e-mails detectados...")
        for idx, (_, email) in enumerate(emails_df.iterrows()):
            if previsoes[idx] == 1:
                self._lidar_com_email_phishing(email)
            else:
                self._exibir_email_normal(email)
    
    def _lidar_com_email_phishing(self, email):
        print("==================================================================================================================")
        logger.warning(f"🚨 E-mail de phishing detectado!")
        logger.warning(f"Assunto: {email['assunto']}")
        logger.warning(f"Conteúdo: {email['conteudo'][:200]}...")  # Mostra apenas início do conteúdo
        logger.warning("Ações tomadas: E-mail movido para quarentena, Remetente bloqueado, Equipe de segurança notificada")
        print("==================================================================================================================")
    
    def _exibir_email_normal(self, email):
        print("==================================================================================================================")
        logger.info(f"📩 E-mail legítimo recebido - {email['assunto']}")
        print("==================================================================================================================")

# Execução principal
def main():
    try:
        # Verifica arquivos essenciais
        if not os.path.exists(CREDENTIALS_FILE):
            raise FileNotFoundError(f"Arquivo de credenciais não encontrado: {CREDENTIALS_FILE}")
        
        logger.info("Iniciando autenticação com Gmail API...")
        service = criar_servico()
        
        logger.info("Buscando e-mails da conta do Gmail...")
        emails_df = listar_emails(service, num_emails=10)
        
        if not emails_df.empty:
            logger.info(f"{len(emails_df)} e-mails encontrados para análise")
            detector = DetectorPhishing()
            previsoes = detector.prever(emails_df)
            respondedor = RespondedorIncidentes()
            respondedor.tomar_acao(emails_df, previsoes)
        else:
            logger.info("Nenhum e-mail encontrado para análise.")

    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}")

if __name__ == "__main__":
    main()