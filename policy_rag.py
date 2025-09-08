import os, re, json, hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from rank_bm25 import BM25Okapi

# Light regex normalization
RE_DEHYPH = re.compile(r"(\w)-\n(\w)")
RE_SPACE_EOL = re.compile(r"[ \t]+\n")
RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
RE_MULTI_BLANK = re.compile(r"\n{3,}")
RE_PAGE_FOOTER = re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.I)
RE_CONFIDENTIAL = re.compile(r"^\s*(confidential|proprietary).*?$", re.I)

def light_normalize(text: str) -> str:
    lines = [ln for ln in text.splitlines()
             if not RE_PAGE_FOOTER.match(ln) and not RE_CONFIDENTIAL.match(ln)]
    t = "\n".join(lines)
    t = RE_DEHYPH.sub(r"\1\2", t)
    t = RE_SPACE_EOL.sub("\n", t)
    t = RE_MULTI_SPACE.sub(" ", t)
    t = RE_MULTI_BLANK.sub("\n\n", t)
    return t.strip()

def load_pdf_light(path: str, source_id: str) -> List[Document]:
    docs = PyMuPDFLoader(path).load()
    for d in docs:
        d.page_content = light_normalize(d.page_content)
        d.metadata.setdefault("source_id", source_id)
        d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
    return docs

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=150,
    separators=["\n## ", "\n# ", "\n\n", "\n", " "],
)

def build_embedder():
    return GoogleGenerativeAIEmbeddings(model="text-embedding-004")

@dataclass
class VectorIndex:
    faiss: FAISS
    bm25_index: BM25Okapi
    chunks: List[Document]

def index_documents(chunks: List[Document]) -> VectorIndex:
    embedder = build_embedder()
    faiss_vs = FAISS.from_documents(chunks, embedder)
    tokenized = [c.page_content.split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return VectorIndex(faiss=faiss_vs, bm25_index=bm25, chunks=chunks)

def hybrid_candidates(index: VectorIndex, query: str, k_dense=20, k_lex=20) -> List[Tuple[Document, float, str]]:
    dense_hits = index.faiss.similarity_search_with_score(query, k=k_dense)
    dense_fmt = [(d, float(s), "dense") for d, s in dense_hits]
    scores = index.bm25_index.get_scores(query.split())
    top_lex_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_lex]
    lex_hits = [(index.chunks[i], float(scores[i]), "bm25") for i in top_lex_idx]

    seen, merged = set(), []
    def key_of(doc: Document):
        sid = doc.metadata.get("source_id", "")
        pg = doc.metadata.get("page", -1)
        h = hashlib.sha1(doc.page_content[:256].encode()).hexdigest()[:8]
        return (sid, pg, h)

    for d, sc, tag in sorted(dense_fmt + lex_hits, key=lambda x: x[1], reverse=True):
        k = key_of(d)
        if k in seen:
            continue
        seen.add(k)
        merged.append((d, sc, tag))
        if len(merged) >= max(k_dense, k_lex):
            break
    return merged

SYSTEM_RULES = (
    "You are a compliance policy assistant.\n"
    "Use only the provided CONTEXT passages.\n"
    "If the answer is not present, reply exactly: Not in policy context.\n"
    "After each sentence, include a citation like [source_id:page].\n"
)

def format_context(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        sid = d.metadata.get("source_id", "src")
        pg = d.metadata.get("page", -1)
        lines.append(f"[CTX{i}] [{sid}:{pg}] {d.page_content[:800]}")
    return "\n".join(lines)

def rerank_with_gemini(candidates: List[Tuple[Document, float, str]], question: str, top_n: int = 8) -> List[Document]:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-pro")
    lines = [f"Q: {question}", 'Rate each passage 0.0-1.0 for answering the question. Return JSON as [{"i":idx,"score":float}].']
    for i, (doc, score, tag) in enumerate(candidates):
        sid = doc.metadata.get("source_id", "src")
        pg = doc.metadata.get("page", -1)
        lines.append(f"#{i} [{sid}:{pg}] {doc.page_content[:800]}")
    prompt = "\n".join(lines)
    resp = model.generate_content(prompt)
    text = resp.text or "[]"
    try:
        scores = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        scored = {int(x["i"]): float(x["score"]) for x in scores if "i" in x and "score" in x}
        ranked = sorted(scored.keys(), key=lambda i: scored[i], reverse=True)
        picked = []
        for i in ranked:
            if 0 <= i < len(candidates):
                picked.append(candidates[i])
                if len(picked) >= top_n:
                    break
    except Exception:
        picked = [candidates[i] for i in range(min(top_n, len(candidates)))]
    return picked

def summarize_and_flag(question: str, context_docs: List[Document]) -> Dict[str, Any]:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = (
        f"{SYSTEM_RULES}\n"
        "Task 1: Answer the question concisely in 3-6 sentences with inline citations.\n"
        "Task 2: Provide a brief summary (2-3 bullets) with citations.\n"
        "Task 3: If anything seems risky/ambiguous/contradictory, output 'fishy' with 1-3 direct quotes and reasons; else 'fishy': [].\n"
        "Return strict JSON with keys: answer, summary (list), fishy (list of {quote, reason, citation}).\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{format_context(context_docs)}\n"
    )
    resp = model.generate_content(prompt)
    txt = resp.text or "{}"
    try:
        data = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    except Exception:
        data = {"answer": "Not in policy context.", "summary": [], "fishy": []}
    data.setdefault("answer", "Not in policy context.")
    data.setdefault("summary", [])
    data.setdefault("fishy", [])
    return data

class PolicyRAG:
    def __init__(self):
        self.index: VectorIndex | None = None

    def ingest_pdf(self, path: str, source_id: str) -> Dict[str, Any]:
        docs = load_pdf_light(path, source_id)
        chunks = splitter.split_documents(docs)
        self.index = index_documents(chunks)
        return {"source_id": source_id, "pages": len(docs), "chunks": len(chunks)}

    def ask(self, question: str, k_dense=20, k_lex=20, top_n=6) -> Dict[str, Any]:
        assert self.index is not None, "Index not built. Ingest a PDF first."
        cands = hybrid_candidates(self.index, question, k_dense=k_dense, k_lex=k_lex)
        ctx = rerank_with_gemini(cands, question, top_n=top_n)
        result = summarize_and_flag(question, ctx)
        result["context"] = [{"source_id": d.metadata.get("source_id"), "page": d.metadata.get("page")} for d in ctx]
        return result
