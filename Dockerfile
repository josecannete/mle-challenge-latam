# syntax=docker/dockerfile:1.2
FROM python:3.13

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD ["fastapi", "run", "--port", "8080", "challenge/api.py"]