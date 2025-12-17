FROM quay.io/jupyter/base-notebook:latest

WORKDIR /home/jovyan

COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./opensearch_index_makar_ozon_semantic.ipynb .
ENV CHOWN_HOME=yes

CMD ["jupyter", "nbconvert", "--execute", "--to", "notebook", "./opensearch_index_makar_ozon_semantic.ipynb"]
