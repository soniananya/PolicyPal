import os, re, json, hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from rank_bm25 import BM25Okapi

# Optional LCEL imports for chain composition
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain  # LCEL retrieval wrapper [LangChain]
# Normalization utilities

RE_DEHYPH = re.compile(r"(\w)-\n(\w)")
RE_SPACE_EOL = re.compile(r"[ \t]+\n")
RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
RE_MULTI_BLANK = re.compile(r"\n{3,}")
RE_PAGE_FOOTER = re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.I)
RE_CONFIDENTIAL = re.compile(r"^\s*(confidential|proprietary).*?$", re.I)

def light_normalize(text: str) -> str:
    lines = [
        ln for ln in text.splitlines()
        if not RE_PAGE_FOOTER.match(ln) and not RE_CONFIDENTIAL.match(ln)
    ]
    t = "\n".join(lines)
    t = RE_DEHYPH.sub(r"\1\2", t)
    t = RE_SPACE_EOL.sub("\n", t)
    t = RE_MULTI_SPACE.sub(" ", t)
    t = RE_MULTI_BLANK.sub("\n\n", t)
    return t.strip()

# PDF loading

def load_pdf_light(path: str, source_id: str) -> List[Document]:
    docs = PyMuPDFLoader(path).load()
    for d in docs:
        d.page_content = light_normalize(d.page_content)
        d.metadata.setdefault("source_id", source_id)
        d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
    return docs  # PyMuPDFLoader and normalization approach aligns with LangChain docs.

# Chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=["\n## ", "\n# ", "\n\n", "\n", " "],
)

# Embeddings

def build_embedder():
    # text-embedding-004 is the current Gemini embeddings model exposed via LangChain.
    return GoogleGenerativeAIEmbeddings(model="text-embedding-004")

# Index

@dataclass
class VectorIndex:
    faiss: FAISS
    bm25_index: BM25Okapi
    chunks: List[Document]

def index_documents(chunks: List[Document]) -> VectorIndex:
    embedder = build_embedder()
    faiss_vs = FAISS.from_documents(chunks, embedder)
    tokenized = [c.page_content.split() for c in chunks]  # build bm25 scorer, for tf,df and all that
    bm25 = BM25Okapi(tokenized)
    return VectorIndex(faiss=faiss_vs, bm25_index=bm25, chunks=chunks)

# Hybrid candidate retrieval

def hybrid_candidates(index: VectorIndex, query: str, k_dense=20, k_lex=20) -> List[Tuple[Document, float, str]]:
    dense_hits = index.faiss.similarity_search_with_score(query, k=k_dense)
    dense_fmt = [(d, float(s), "dense") for d, s in dense_hits]
    scores = index.bm25_index.get_scores(query.split())
    top_lex_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_lex]    # bm25 returns an arr so sort and choose top k
    lex_hits = [(index.chunks[i], float(scores[i]), "bm25") for i in top_lex_idx]

    seen, merged = set(), []

    def key_of(doc: Document):
        sid = doc.metadata.get("source_id", "")
        pg = doc.metadata.get("page", -1)
        h = hashlib.sha1(doc.page_content[:256].encode()).hexdigest()[:8]
        return (sid, pg, h)   # to avoid duplicates, use source_id, page, and a hash of the content(first 256 chars) as a key, so if something repeats, we know

    for d, sc, tag in sorted(dense_fmt + lex_hits, key=lambda x: x, reverse=True):  # combine and sort descending by score
        k = key_of(d) # create a key for the doc
        if k in seen:
            continue
        seen.add(k)
        merged.append((d, sc, tag))
        if len(merged) >= max(k_dense, k_lex):
            break
    return merged

# System rules and context

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

# Gemini rerank (LangChain-friendly, still simple JSON, no regex)

def rerank_with_gemini(candidates, question, top_n=8):
    # Setup
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    # Use LangChain chat wrapper to keep the stack consistent. [12]
    chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    # Build prompt text
    passages = []
    for i, (doc, _, _) in enumerate(candidates):
        sid = doc.metadata.get("source_id", "src")
        pg = doc.metadata.get("page", -1)
        passages.append({"i": i, "sid": sid, "pg": pg, "text": doc.page_content[:800]})

    prompt_text = (
        f"Q: {question}\n"
        "Score each passage 0.0–1.0 for usefulness in answering Q.\n"
        'Return ONLY a JSON array of {"i": int, "score": float} objects.\n\n' +
        "\n".join(f"#{p['i']} [{p['sid']}:{p['pg']}] {p['text']}" for p in passages)
    )

    # Ask Gemini; prefer structured output if available later; here, parse JSON directly (no regex). [9][12]
    resp = chat.invoke(prompt_text)
    text = resp.content or "[]"

    try:
        arr = json.loads(text)
        idx2score = {int(x["i"]): float(x["score"]) for x in arr if "i" in x and "score" in x}
        top_idx = sorted(idx2score, key=idx2score.get, reverse=True)[:top_n]
    except Exception:
        top_idx = list(range(min(top_n, len(candidates))))

    return [candidates[i] for i in top_idx]

# Summarize + flag (LangChain prompt template)

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_rules}"),
    ("human",
     "Task 1: Answer the question concisely in 3-6 sentences with inline citations.\n"
     "Task 2: Provide a brief summary (2-3 bullets) with citations.\n"
     "Task 3: If anything seems risky/ambiguous/contradictory, output 'fishy' with 1-3 direct quotes and reasons; else 'fishy': [].\n"
     "Return strict JSON with keys: answer, summary (list), fishy (list of {quote, reason, citation}).\n\n"
     "QUESTION:\n{question}\n\n"
     "CONTEXT:\n{context}\n")
])


_summary_chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro") 

def summarize_and_flag(question: str, context_docs: List[Document]) -> Dict[str, Any]:
    context = format_context(context_docs)
    messages = summary_prompt.format_messages(
        system_rules=SYSTEM_RULES,
        question=question,
        context=context
    )
    resp = _summary_chat.invoke(messages)
    txt = resp.content or "{}"
    try:
        data = json.loads(txt)  # no regex; relies on model producing clean JSON
    except Exception:
        data = {"answer": "Not in policy context.", "summary": [], "fishy": []}
    data.setdefault("answer", "Not in policy context.")
    data.setdefault("summary", [])
    data.setdefault("fishy", [])
    return data
# Structured outputs via with_structured_output can further harden JSON parsing if desired. [9][18]

class PolicyRAG:
    def __init__(self):
        self.index: VectorIndex | None = None

    def ingest_pdf(self, path: str, source_id: str) -> Dict[str, Any]:
        docs = load_pdf_light(path, source_id)
        chunks = splitter.split_documents(docs)
        self.index = index_documents(chunks)
        return {"source_id": source_id, "pages": len(docs), "chunks": len(chunks)}
        # Matches typical LangChain ingest + split + vectorize flow.

    def ask(self, question: str, k_dense=20, k_lex=20, top_n=6) -> Dict[str, Any]:
        assert self.index is not None, "Index not built. Ingest a PDF first."
        cands = hybrid_candidates(self.index, question, k_dense=k_dense, k_lex=k_lex)
        ctx_docs = rerank_with_gemini(cands, question, top_n=top_n)
        result = summarize_and_flag(question, ctx_docs)
        result["context"] = [
            {"source_id": d.metadata.get("source_id"), "page": d.metadata.get("page")}
            for d in ctx_docs
        ]
        return result
        # Hybrid retrieval + rerank + constrained answer generation is consistent with LangChain hybrid patterns.

