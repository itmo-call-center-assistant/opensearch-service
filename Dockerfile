FROM python:3.14-slim-trixie

WORKDIR /app

# install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN ["pip", "install", "--no-cache-dir", "--upgrade", "-r", "/app/requirements.txt"]

COPY ./scripts /app/scripts

EXPOSE 8000
CMD ["uvicorn", "scripts.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
