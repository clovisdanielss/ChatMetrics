FROM python:3.8.12
WORKDIR /app
COPY requirements.txt .
COPY . .
RUN pip install -r requirements.txt
RUN python -m spacy download pt_core_news_lg
CMD ["python", "./__init__.py"]

EXPOSE 5000
EXPOSE 443