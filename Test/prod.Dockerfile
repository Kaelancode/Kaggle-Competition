FROM python:3.8-slim-buster

RUN pip install --upgrade pip

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .

CMD ["jupyter", "notebook", "--port=8888", "--ip=0.0.0.0", "--allow-root"]
