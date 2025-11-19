import os
import re
import uvicorn
import logging
import traceback
from app import services
from datetime import datetime
from datetime import timezone
from pydantic import BaseModel
from dotenv import load_dotenv
from db.auth import verify_token
from db.auth import conditional_auth

from pymongo import MongoClient
from db.engine import MONGO_URI, MONGO_DB
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from intent_classifier import IntentClassifier
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Depends

logger = logging.getLogger(__name__)

# Carregando variáveis de ambiente do .env
load_dotenv()

# Lendo o ambiente (dev ou prod)
ENV = os.getenv("ENV", "prod").lower()
logger.info(f"Running in {ENV} mode")

MODELS = {}

def get_model_urls() -> str:
    """
    Busca a string de URLs de modelos da variável de ambiente WANDB_MODELS.
    Isolar essa lógica em uma função facilita o patching durante os testes.
    """
    confusion_url = os.getenv("WANDB_CONFUSION_MODEL_URL")
    clair_url = os.getenv("WANDB_CLAIR_MODEL_URL")
    if not confusion_url or not clair_url:
        raise ValueError("URLs dos modelos (CONFUSION ou CLAIR) não estão definidas")
    
    return f"{confusion_url},{clair_url}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Inicialização do app. Atualmente, apenas carrega modelos do W&B.
    """
    global MODELS
    logger.info("Carregando modelos do W&B durante a inicialização do app...")
    try:
        model_urls_str = get_model_urls()
        MODELS = services.load_all_classifiers(model_urls_str)
        logger.info("Modelos do W&B carregados com sucesso.")
    except Exception as e:
        logger.error(f"Falha crítica ao carregar modelos do W&B: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Falha crítica ao carregar modelos do W&B: {str(e)}")
    # This is the point where the app is ready to handle requests
    yield
    # Código para ser executado no shutdown (opcional)
    logger.info("Descarregando modelos e limpando recursos...")
    MODELS.clear()


# Inicializando a aplicação FastAPI
app = FastAPI(
    title="Aplicação Básica de ML",
    description="Uma API simples para classificar intenções usando modelos de ML pré-treinados.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configurando CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",  # React ou outra frontend local
        "https://meusite.com",    # domínio em produção
    ],
    allow_credentials=True,
    allow_methods=["*"],              # permite todos os métodos: GET, POST, etc
    allow_headers=["*"],              # permite todos os headers (Authorization, Content-Type...)
)

"""
Routes
"""
@app.get("/")
async def root():
    return {"message": f"Aplicação Básica de ML está executando no modo {ENV}."}

@app.post("/predict")
async def predict(text: str, owner: str = Depends(conditional_auth)):
    """
    Endpoint de predição.
    Este é um 'Controller' enxuto. 
    Ele apenas delega a lógica de negócio para o services.py.
    """
    try:
        # 1. O Controller delega TODA a lógica de negócio para o services.py
        results = services.predict_and_log_intent(
            text=text, 
            owner=owner, 
            models=MODELS
        )
        # 2. O Controller retorna a resposta (Lógica de View) no formato JSON
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Erro ao processar a predição: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a predição: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)