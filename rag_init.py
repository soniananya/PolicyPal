import json
from db import SessionLocal
from db_io import insert_policy, insert_policy_text, insert_query, get_policy_owned
from policy_rag import PolicyRAG

class RAGService:
    def __init__(self):
        self.rag = PolicyRAG()

    def ingest_pdf_and_persist(self, *, user_id: str, policy_name: str, policy_type: str,
                               storage_bucket: str, storage_path: str, local_pdf_path: str,
                               extra_metadata: dict) -> dict:
        with SessionLocal() as session, session.begin():
            policy_id = insert_policy(session,
                user_id=user_id, policy_name=policy_name, policy_type=policy_type,
                storage_bucket=storage_bucket, storage_path=storage_path, metadata=extra_metadata or {}
            )
        stats = self.rag.ingest_pdf(local_pdf_path, source_id=policy_id)
        raw_text = ""  # optional snapshot
        with SessionLocal() as session, session.begin():
            insert_policy_text(session, policy_id=policy_id, raw_text=raw_text)
        return {"policy_id": policy_id, **stats}

    def answer_and_persist(self, *, user_id: str, policy_id: str, question: str) -> dict:
        with SessionLocal() as session:
            if not get_policy_owned(session, user_id=user_id, policy_id=policy_id):
                raise PermissionError("Policy not found for this user")
        out = self.rag.ask(question)
        payload = {
            "answer": out.get("answer", ""),
            "summary": out.get("summary", []),
            "fishy": out.get("fishy", []),
            "context": out.get("context", []),
        }
        answer_text = payload["answer"].strip() + "\n\n" + json.dumps(payload, ensure_ascii=False)
        with SessionLocal() as session, session.begin():
            qid = insert_query(session, user_id=user_id, policy_id=policy_id, query_text=question, answer_text=answer_text)
        payload["query_id"] = qid
        return payload
