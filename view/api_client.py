"""
Cliente da API.
Este módulo lida com toda a comunicação com a API FastAPI.
"""

import requests
from view.config import API_URL

class APIConnectionError(Exception):
    """Exceção para erros de conexão com a API."""
    pass

class APIError(Exception):
    """Exceção para erros de resposta da API (status != 200)."""
    pass

def fetch_prediction(text: str) -> dict:
    """
    Chama a API de predição e retorna o resultado em JSON.

    Args:
        text: O texto a ser classificado.

    Returns:
        Um dicionário com a resposta da API.

    Raises:
        APIConnectionError: Se não for possível conectar à API.
        APIError: Se a API retornar um status de erro (não-200).
    """
    if not text:
        return {}

    params = {"text": text}
    
    try:
        # Chama a API com os parâmetros
        response = requests.post(API_URL, params=params)

        # Levanta um erro HTTP para respostas ruins (4xx, 5xx)
        response.raise_for_status() 

        # Retorna o JSON se tudo deu certo
        return response.json()

    except requests.exceptions.ConnectionError as e:
        # "Traduz" o erro de requests para nossa exceção customizada
        raise APIConnectionError(
            f"Não foi possível conectar à API. "
            f"Verifique se ela está rodando em {API_URL}."
        ) from e
        
    except requests.exceptions.HTTPError as e:
        # "Traduz" o erro de status
        raise APIError(
            f"Falha ao chamar a API. Status: {e.response.status_code}. "
            f"Resposta: {e.response.text}"
        ) from e