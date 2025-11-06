FROM python:3.11-slim-bullseye

# Diretório dentro do contêiner
WORKDIR /app

# Criar um usuário não-root para rodar a aplicação
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app

# Instalando dependências necessárias do sistema
RUN apt-get update && apt-get install -y build-essential libffi-dev && rm -rf /var/lib/apt/lists/*

# Mudando para o usuário não-root
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

# Versão mais recente do pip
RUN pip install --no-cache-dir --upgrade pip

# Copiando o arquivo de requisitos
COPY --chown=appuser:appuser requirements.txt .

# Instalando dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiando o código da aplicação
COPY --chown=appuser:appuser . .

# Expondo a porta em que a aplicação irá rodar
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]