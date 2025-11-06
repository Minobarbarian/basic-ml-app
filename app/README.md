# App


## Comando para rodar a aplicação
```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --log-level debug
```
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]

## Criar um novo token
``` bash
python app/auth.py create --owner="alguem" --expires_in_days=365
```

## Ler todos os tokens
``` bash
python app/auth.py read_all
```