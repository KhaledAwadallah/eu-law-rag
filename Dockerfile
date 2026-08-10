# AskArxiv web app. The vector index is NOT baked into the image;
# mount it at runtime:  docker run -v ./data:/app/data ...
FROM python:3.12-slim

WORKDIR /app

# Dependencies first: this layer is cached and only rebuilds when
# requirements.txt changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then the code (changes often, but the layer above stays cached).
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .
COPY app.py .

# Gradio must listen on all interfaces inside a container, not just localhost.
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860

CMD ["python", "app.py"]
