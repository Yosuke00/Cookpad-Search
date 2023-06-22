FROM python:3.9-buster
ENV PYTHONUNBUFFERED=1

WORKDIR /app/

RUN pip install poetry

COPY pyproject.toml* poetry.lock* ./

RUN poetry config virtualenvs.in-project true
RUN if [ -f pyproject.toml ]; then poetry install --no-root; fi

COPY ./api ./

COPY ./templates ./

ENTRYPOINT ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--reload"]
