
#python3.9のイメージをダウンロード
FROM python:3.9-buster
ENV PYTHONUNBUFFERED=1

WORKDIR /app/

#pipを使ってpoetryをインストール
RUN pip install poetry

#poetryの定義ファイルをコピー(存在する場合)
COPY pyproject.toml* poetry.lock* ./

#poetryでライブラリをインストール(pyproject.tomlが既にある場合)
RUN poetry config virtualenvs.in-project true
RUN if [ -f pyproject.toml ]; then poetry install --no-root; fi

#API部分のCOPY
COPY ./api ./
#UI部分のコピー
COPY ./templates ./

#uvicornのサーバーを立ち上げる
ENTRYPOINT ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--reload"]
