import os
import re
import uvicorn
import logging
import traceback
from datetime import datetime
from datetime import timezone
from pydantic import BaseModel
from dotenv import load_dotenv
from app.auth import verify_token
from db.engine import get_mongo_collection
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

# Inicializando a aplicação FastAPI
app = FastAPI(
    title="Aplicação Básica de ML",
    description="Uma API simples para classificar intenções usando modelos de ML pré-treinados.",
    version="1.0.0",
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # ReDoc
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

# Inicializando conexão com MongoDB
try:
    collection = get_mongo_collection(f"{ENV.upper()}_intent_logs")
    logger.info("Conexão estabelecida com MongoDB.")
except Exception as e:
    logger.error(f"Erro ao conectar com MongoDB:{str(e)}")
    logger.error(traceback.format_exc())

async def conditional_auth(request: Request):
    global ENV
    if ENV == "dev":
        logger.info("Modo DEV: pulando autenticação.")
        return "dev_user"
    else:
        try:
            return verify_token(request)
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Erro na autenticação: {str(e)}")
            raise HTTPException(status_code=401, detail="Autenticação falhou.")

# Carregando os modelos
MODELOS = {}
try:
    logger.info("Carregando modelos...")
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, "..", "intent_classifier", "models")
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = model_file.replace(".keras", "").upper()
        MODELOS[model_name] = IntentClassifier(load_model=model_path)
        logger.info(f"Modelo '{model_name}' carregado de '{model_path}'")
except Exception as e:
    logger.error(f"Erro ao carregar modelos: {str(e)}")
    logger.error(traceback.format_exc())

"""
Routes
"""
@app.get("/")
async def root():
    return {"message": f"Aplicação Básica de ML está executando no modo {ENV}."}

@app.post("/predict")
async def predict(text: str, owner: str = Depends(conditional_auth)):
    # Gerando predições com todos os modelos
    predictions = {}
    for model_name, model in MODELOS.items():
        top_intent, all_probs = model.predict(text)
        predictions[model_name] = {
            "top_intent": top_intent,
            "all_probs": all_probs
        }

    results = {
        "text": text, 
        "owner": owner, 
        "predictions": predictions, 
        "timestamp": int(datetime.now(timezone.utc).timestamp())
    }
    
    collection.insert_one(results)
    results['id'] = str(results['_id'])
    results.pop('_id')

    return JSONResponse(content=results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)