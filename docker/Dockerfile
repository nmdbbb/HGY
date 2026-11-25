FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY configs ./configs

CMD ["uvicorn", "truyen_reco.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
