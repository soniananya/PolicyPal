from typing import List, Optional, Dict, Any
import os, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from db import init_models
from rag_init import RAGService

init_models()
app = FastAPI()
rag = RAGService()

class IngestMeta(BaseModel):
    checksum: Optional[str] = None
    status: Optional[str] = "uploaded"
    extra: Dict[str, Any] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    policy_id: str
    source_id: str
    pages: int
    chunks: int

class QueryRequest(BaseModel):
    user_id: str
    policy_id: str
    question: str
    top_n: int = 6

class Evidence(BaseModel):
    source_id: str
    page: int

class QueryResponse(BaseModel):
    query_id: str
    answer: str
    summary: List[str]
    fishy: List[Dict[str, Any]]
    context: List[Evidence]

# endpoints

@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    user_id: str = Form(...),
    policy_name: str = Form(...),
    policy_type: str = Form(""),
    storage_bucket: str = Form("local"),
    storage_path: str = Form(""),
    metadata_json: str = Form("{}"),
    file: UploadFile = File(...),
):
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        meta = IngestMeta(**(json.loads(metadata_json) if metadata_json else {}))
    except Exception:
        meta = IngestMeta()

    if not storage_path:
        storage_path = file.filename

    out = rag.ingest_pdf_and_persist(
        user_id=user_id,
        policy_name=policy_name,
        policy_type=policy_type,
        storage_bucket=storage_bucket,
        storage_path=storage_path,
        local_pdf_path=tmp_path,
        extra_metadata={"checksum": meta.checksum, "status": meta.status, **meta.extra},
    )

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return IngestResponse(
        policy_id=out["policy_id"],
        source_id=out["source_id"],
        pages=out["pages"],
        chunks=out["chunks"],
    )

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        out = rag.answer_and_persist(user_id=req.user_id, policy_id=req.policy_id, question=req.question)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    return QueryResponse(
        query_id=out["query_id"],
        answer=out["answer"],
        summary=out["summary"],
        fishy=out["fishy"],
        context=[Evidence(**c) for c in out["context"]],
    )
