# Reproducible environment for the equity screener: Linux, Python 3.12, and the exact package
# versions the test suite was verified with (requirements-tested.txt).
#
#   docker build -t eqscreen .
#   docker run --rm -v "${PWD}/outputs:/app/outputs" eqscreen                 # offline demo
#   docker run --rm --env-file .env -v "${PWD}/data:/app/data" -v "${PWD}/data_raw:/app/data_raw" \
#       -v "${PWD}/outputs:/app/outputs" -v "${PWD}/logs:/app/logs" eqscreen --mode live
#   docker run --rm --entrypoint python eqscreen -m pytest                    # test suite
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-tested.txt ./
RUN pip install -r requirements-tested.txt

COPY . .
RUN pip install --no-deps -e .

# Mount these so the database, downloads, results and logs outlive the container.
VOLUME ["/app/data", "/app/data_raw", "/app/outputs", "/app/logs"]

ENTRYPOINT ["python", "-m", "eqscreen.pipeline"]
CMD ["--mode", "demo"]
