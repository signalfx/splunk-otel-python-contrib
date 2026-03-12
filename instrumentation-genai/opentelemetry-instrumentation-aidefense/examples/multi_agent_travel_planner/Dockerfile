FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install local AI Defense instrumentor (overrides the published package)
COPY aidefense_pkg/ aidefense_pkg/
RUN pip install --no-cache-dir aidefense_pkg/

COPY main.py .
COPY util/ util/

USER 1000

CMD ["python", "main.py"]
