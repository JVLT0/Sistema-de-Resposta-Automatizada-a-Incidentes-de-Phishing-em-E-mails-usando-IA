from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# Escopo para leitura de e-mails
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# --- Caminhos ajustados ---
# Base: script está em `src/auth/gmailauth.py`
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Volta até a raiz do projeto
CONFIG_DIR = os.path.join(BASE_DIR, 'config')  # Pasta config/ no nível raiz

CREDENTIALS_FILE = os.path.join(CONFIG_DIR, 'client_secret.json')  # ou credentials.json
TOKEN_FILE = os.path.join(CONFIG_DIR, 'token.json')

# Verifica se os arquivos existem
if not os.path.exists(CREDENTIALS_FILE):
    raise FileNotFoundError(f"🔴 Arquivo de credenciais não encontrado: {CREDENTIALS_FILE}")
print(f"🟢 Credenciais carregadas de: {CREDENTIALS_FILE}")

# Cria a pasta config/ se não existir
os.makedirs(CONFIG_DIR, exist_ok=True)

# --- Função de autenticação ---
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
        
        # Salva o token atualizado
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

# Execução
try:
    service = criar_servico()
    print("🟢 Serviço Gmail criado com sucesso!")
except Exception as e:
    print(f"🔴 Erro: {e}")