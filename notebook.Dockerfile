FROM quay.io/jupyter/base-notebook:latest

WORKDIR /app

COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./opensearch_index_makar_ozon_semantic.ipynb .
COPY ./docs ./docs/

CMD ["jupyter", "nbconvert", "--execute", "--to", "notebook", "--inplace", "opensearch_index_makar_ozon_semantic.ipynb"]
