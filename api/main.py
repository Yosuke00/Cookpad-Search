from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import aiohttp
from api.model import SearchModel
import uvicorn

#FastAPIのインスタンス化
app = FastAPI()
#UI部分のテンプレートの指示
templates = Jinja2Templates(directory='templates')
# 類似文章検索エンジンAPIのエンドポイント
model = SearchModel()

#表示画面
@app.get("/", response_class=HTMLResponse)
def home(query:str, request: Request):
    return templates.TemplateResponse("index.html", {"request": request,'query':query})

#クエリの入力が行われると結果をPOSTするメソッド
@app.post('/result/{query}')
async def semantic_search(query:str, top_k:int):
    results = model.get_result(query, top_k)  # 検索結果を取得
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)