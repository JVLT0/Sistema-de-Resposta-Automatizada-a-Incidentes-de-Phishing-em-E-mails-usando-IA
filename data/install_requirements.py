import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Lista de dependências do projeto
dependencias = [
    "numpy",
    "pandas",
    "torch",
    "transformers",
    "nltk",
    "google-auth-oauthlib",
    "google-auth",
    "google-auth-httplib2",
    "google-api-python-client",
    "beautifulsoup4",
    "requests"
]

# Instala todas as dependências
for pacote in dependencias:
    print(f"Instalando {pacote}...")
    install(pacote)

print("✅ Instalação concluída!")