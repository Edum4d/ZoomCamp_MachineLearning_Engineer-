FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.profile
ENV PATH="/root/.local/bin:${PATH}"


COPY app.py pipeline_v1.bin ./

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uv","run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]