FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ROOT=/app

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY pyproject.toml ./
COPY app ./app
COPY modeldata/house_price_dataset.csv modeldata/test_data_for_prediction.csv ./modeldata/

RUN python -m pip install --no-cache-dir . \
    && python -m app.training \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]