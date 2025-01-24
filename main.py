from fastapi import FastAPI
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

templates = Jinja2Templates(directory="templates")


app = FastAPI(docs_url="/documentation", redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("main_page.html",{"request":request})

@app.get("/asdf")
async def home(request: Request):
    return templates.TemplateResponse("test.html",{"request":request})


#http://localhost:8000