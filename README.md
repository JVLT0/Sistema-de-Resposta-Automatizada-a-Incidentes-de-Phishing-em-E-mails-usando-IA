# 🛡️ Sistema de Resposta Automatizada a Incidentes de Phishing em E-mails usando IA
## 📌 Descrição do Projeto
Este projeto utiliza Inteligência Artificial (BERT em português) para detectar e responder automaticamente a e-mails de phishing. O sistema analisa o conteúdo e o assunto de e-mails em tempo real, classifica-os como legítimos ou maliciosos e toma ações automatizadas (como mover para quarentena ou notificar equipes de segurança).

## 🎯 Objetivos  
- 🔍 Detectar phishing em e-mails usando modelos de NLP (BERT).
- ⚡ Automatizar respostas a incidentes de segurança.
- 📊 Gerar relatórios de e-mails maliciosos identificados.
- 🛠️ Integrar com APIs de e-mail (Gmail, Outlook).
- 🔄 Aprender continuamente com novos padrões de ataques.

## 🛠️ Principais Funcionalidades  
### **Classificação de E-mails:**
- ✅ Legítimos: Exibidos normalmente.
- ❌ Phishing: Bloqueados e reportados.

### **Ações Automatizadas:**
- 🚨 Quarentena de e-mails maliciosos.
- 📧 Notificação para equipes de segurança.

# 🔧 Tecnologias Utilizadas
- 🤖 IA/NLP:
- BERT em português (neuralmind/bert-base-portuguese-cased).
- PyTorch para treinamento do modelo.

# 📧 APIs de E-mail:
- Gmail API (para integração com contas corporativas).

# 🛠️ Infraestrutura:
- Python 3.9+.
- FastAPI (para endpoints de detecção).
- Docker (para deploy).

# 🚀 Como Executar
### **Pré-requisitos:**
- Python 3.9+ e pip instalados.
- Credenciais da API do Gmail (veja config/credentials.json).

### **Instalação:**  
```bash
git clone https://github.com/JVLT0/Sistema-de-Resposta-Automatizada-a-Incidentes-de-Phishing-em-E-mails-usando-IA
cd projeto
```
- Instale os requerimentos para o projeto: 
   ```bash 
   data/install_requirements.py 
   ```

### **Configuração:**
- Adicione suas credenciais do Gmail em:
    ```bash
    config/client_secret.json
    ```

- Execute o treinamento do modelo (opcional):
    ```bash
    Copy python src/ai/treinamento_bert.py
    ```

### **Iniciar o Sistema:**
```bash
python src/main.py
```

# 📌 Estrutura do Projeto
projeto-phishing/  
├── config/               # Credenciais e tokens (NÃO versionados)  
├── data/                 # Datasets para treinamento (ex: dataset_emails.csv)  
├── models/               # Modelos BERT treinados (ex: modelo_bert_phishing.pth)  
├── src/  
│   ├── ai/               # Código de IA (treinamento, inferência)  
│   ├── auth/             # Autenticação com APIs de e-mail  
│   └── main.py           # Ponto de entrada do sistema  
└── README.md  

# 📊 Métricas de Desempenho
- Acurácia do Modelo: 95% (em testes com dataset balanceado).
- Tempo de Resposta: < 2 segundos por e-mail.

# 🔄 Próximos Passos
- 🔄 Adicionar suporte a mais provedores de e-mail (Outlook, Exchange).
- 🧠 Implementar fine-tuning contínuo do modelo com novos dados.
- 🌐 Desenvolver um dashboard web para monitoramento.