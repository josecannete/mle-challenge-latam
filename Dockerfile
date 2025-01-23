# syntax=docker/dockerfile:1.2
FROM python:3.13

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir .

CMD ["fastapi", "run", "challenge/api.py"]