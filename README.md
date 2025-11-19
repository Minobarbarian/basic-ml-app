# ML-APP
Este repositório foi copiado de https://github.com/adaj/basic-ml-app e adaptado para usar dois modelos de classificadores de intenção (confusion e clair).

## 🏛️ Estrutura atual do projeto

```shell
.                               # "Working directory"
├── app/                        # Lógica do serviço web
│   ├── app.py                  # Implementação do backend com FastAPI
│   ├── app.Dockerfile          # Definição do container em que o backend roda
│   ├── schema.py               
│   └── services.py               
├── db/                         # Lógica do banco de dados
│   ├── auth.py
│   ├── engine.py               # Encapsulamento do pymongo
│   └── test.py               
├── intent-classifier/          # Scripts relacionados ao modelo de ML
│   ├── data/                   # Dados para os modelos de ML
│   ├── models/                 # Modelos treinados
│   └── intent-classifier.py    # Código principal do modelo de ML
├── dags/                       # Workflows integrados no Airflow
│   └── ...                     # TODO
├── tests/                      # Testes unitários e de integração
│   ├── test_app.py
│   └── test_intent_classifier.py
├── docker-compose.yml          # Arquivo de orquestração dos serviços envolvidos
├── requirements.txt            # Dependências do Python
├── .env                        # Variáveis de ambiente
└── .gitignore
```
## ⚙️ Instruções para deploy em ambiente de teste

### Localmente
#### Para o backend, num terminal:
```shell
# Crie e ative um ambiente conda com as dependências do projeto
conda create -n intent-clf python=3.11
conda activate intent-clf
pip install -r requirements.txt # instalar as dependências
## Ajuste seu .env com as variáveis de ambiente necessárias
export ENV=dev
## Em .env, se ENV=prod, você precisará criar um token
## O IP da máquina precisa ser permitido no MongoDB também
python -m app.auth create --owner="nome" --expires_in_days=365
# Suba o serviço web e acesse-o em localhost:8000
uvicorn app.app:app --host 0.0.0.0 --port 8000 --log-level debug
```

#### Para o frontend, noutro terminal:
```shell
conda activate intent-clf
python -m streamlit run view/streamlit_app.py
```
Quando estiver executando, acesse o link fornecido nesse terminal.

### Utilizando o Docker

### Construindo a imagem do container
```shell
sudo docker build -t intent-clf:0.1 -f app/app.Dockerfile .
```

### Executando o container 
```shell
sudo docker run -d -p 8080:8000 --name intent-clf-container intent-clf:0.1
# Checar os containers ativos
sudo docker ps
# Acompanhar os logs do container
sudo docker logs -f intent-clf-container
```
Ou construa um arquivo `docker-compose.yml` (útil para execução de vários containers com um só comando) e execute:
```shell
sudo docker-compose up -d
# Checar os containers ativos
sudo docker ps
# Acompanhar os logs do container
sudo docker logs -f intent-clf-container
```
Para interromper a execução do container:
```shell
# Parar o container
sudo docker stop intent-clf-container
# Deletar o container (com -f ou --force você deleta sem precisar parar)
sudo docker rm -f intent-clf-container
```