import uuid
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from db import Policy, PolicyText, Query, FeedbackEnum

def new_id() -> str:
    return uuid.uuid4().hex

def insert_policy(session: Session, *, user_id: str, policy_name: str,
                  policy_type: str, storage_bucket: str, storage_path: str,
                  metadata: Dict[str, Any]) -> str:
    pid = new_id()
    row = Policy(
        policy_id=pid, user_id=user_id, 
        policy_name=policy_name,
        policy_type=policy_type, 
        storage_bucket=storage_bucket,
        storage_path=storage_path, 
        metadata=metadata or {},
    )
    session.add(row)
    session.flush()
    return pid

def insert_policy_text(session: Session, *, policy_id: str, raw_text: str) -> str:
    ptid = new_id()
    row = PolicyText(policy_text_id=ptid, 
           policy_id=policy_id, 
           raw_text=raw_text)
    session.add(row)
    session.flush()
    return ptid

def insert_query(session: Session, *, user_id: str, policy_id: str,
                 query_text: str, answer_text: str) -> str:
    qid = new_id()
    row = Query(
        query_id=qid, 
        user_id=user_id, 
        policy_id=policy_id,
        query_text=query_text, 
        answer_text=answer_text,
        feedback=FeedbackEnum.none,
    )
    session.add(row)
    session.flush()
    return qid

def get_policy_owned(session: Session, *, user_id: str, policy_id: str) -> Optional[Policy]:
    return session.query(Policy).filter(
        Policy.policy_id == policy_id, Policy.user_id == user_id
    ).first()
