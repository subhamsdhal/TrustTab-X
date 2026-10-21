FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY pyproject.toml README.md ./
COPY constraints ./constraints
COPY src ./src
RUN pip install --no-cache-dir -c constraints/m5.txt '.[m5]' && useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1
CMD ["uvicorn", "trusttabx.m5_api:app", "--host", "0.0.0.0", "--port", "8000"]
