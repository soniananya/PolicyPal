import os, json, tempfile
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from db import init_models
from rag_init import RAGService

init_models()
app = FastAPI(title="PolicyPal RAG")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

rag = RAGService()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "ingest": None, "answer": None, "error": None})

@app.post("/ingest_ui", response_class=HTMLResponse)
async def ingest_ui(
    request: Request,
    user_id: str = Form(...),
    policy_name: str = Form(...),
    policy_type: str = Form(""),
    storage_bucket: str = Form("local"),
    storage_path: str = Form(""),
    metadata_json: str = Form("{}"),
    file: UploadFile = File(...),
):
    # Save temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        meta = json.loads(metadata_json) if metadata_json else {}
    except Exception:
        meta = {}

    if not storage_path:
        storage_path = file.filename

    try:
        out = rag.ingest_pdf_and_persist(
            user_id=user_id,
            policy_name=policy_name,
            policy_type=policy_type,
            storage_bucket=storage_bucket,
            storage_path=storage_path,
            local_pdf_path=tmp_path,
            extra_metadata=meta,
        )
    except Exception as e:
        os.unlink(tmp_path)
        return templates.TemplateResponse("index.html", {"request": request, "ingest": None, "answer": None, "error": str(e)})

    os.unlink(tmp_path)
    return templates.TemplateResponse("index.html", {"request": request, "ingest": out, "answer": None, "error": None})

@app.post("/query_ui", response_class=HTMLResponse)
async def query_ui(request: Request, user_id: str = Form(...), policy_id: str = Form(...), question: str = Form(...)):
    try:
        out = rag.answer_and_persist(user_id=user_id, policy_id=policy_id, question=question)
    except PermissionError as e:
        return templates.TemplateResponse("index.html", {"request": request, "ingest": None, "answer": None, "error": str(e)})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "ingest": None, "answer": None, "error": str(e)})

    return templates.TemplateResponse("index.html", {"request": request, "ingest": None, "answer": out, "error": None})
