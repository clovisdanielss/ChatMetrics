FROM python:3.8.12
WORKDIR /app
COPY requirements.txt .
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "./__init__.py"]

EXPOSE 5000
EXPOSE 443