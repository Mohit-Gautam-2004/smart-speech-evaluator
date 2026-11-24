"""MongoDB database manager for Smart Speak.

Provides functionality for:
1. Personal Reference Library (saving & loading topics/knowledge bases).
2. Practice Attempts & Progress Tracking (storing attempts and calculating score deltas).
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
except ImportError:
    MongoClient = None
    Collection = None
    Database = None

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

DEFAULT_DB_NAME = os.getenv("MONGODB_DB_NAME", "smart_speak")


def clean_uri(raw_uri: str) -> str:
    """Sanitize MongoDB URI by removing accidental < and > around the password."""
    import re
    cleaned = raw_uri.strip()
    return re.sub(r":<([^>]+)>@", r":\1@", cleaned)


def get_client(uri: str | None = None) -> MongoClient | None:
    """Create a MongoClient with a quick timeout to prevent UI freezes."""
    if MongoClient is None:
        return None
    load_dotenv(dotenv_path=env_path, override=True)
    mongo_uri = clean_uri(uri or os.getenv("MONGODB_URI", ""))
    if not mongo_uri:
        return None
    try:
        return MongoClient(mongo_uri, serverSelectionTimeoutMS=4000)
    except Exception:
        return None


def get_db(uri: str | None = None, db_name: str | None = None) -> Database | None:
    """Get the MongoDB database instance."""
    client = get_client(uri)
    if client is None:
        return None
    load_dotenv(dotenv_path=env_path, override=True)
    name = (db_name or os.getenv("MONGODB_DB_NAME", DEFAULT_DB_NAME)).strip() or DEFAULT_DB_NAME
    return client[name]


def test_connection(uri: str | None = None) -> tuple[bool, str]:
    """Test connection to MongoDB."""
    if MongoClient is None:
        return False, "pymongo package is not installed."
    load_dotenv(dotenv_path=env_path, override=True)
    mongo_uri = clean_uri(uri or os.getenv("MONGODB_URI", ""))
    if not mongo_uri:
        return False, "MONGODB_URI is not configured in .env."
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return True, "Connected successfully to MongoDB."
    except Exception as exc:
        return False, f"Connection failed: {exc}"


# ==========================================================
# 1. Personal Reference Library
# ==========================================================

def save_reference_topic(
    user_id: str,
    topic_name: str,
    expected_content: str,
    expected_tone: str,
    knowledge_text: str,
    uri: str | None = None,
) -> bool:
    """Save or update a reference topic in the user's personal library."""
    db = get_db(uri)
    if db is None:
        return False

    col = db["reference_library"]
    now = datetime.datetime.now(datetime.timezone.utc)
    clean_topic = topic_name.strip()
    clean_user = (user_id or "default_user").strip()

    col.update_one(
        {"user_id": clean_user, "topic_name": clean_topic},
        {
            "$set": {
                "user_id": clean_user,
                "topic_name": clean_topic,
                "expected_content": expected_content.strip(),
                "expected_tone": expected_tone.strip(),
                "knowledge_text": knowledge_text.strip(),
                "updated_at": now,
            },
            "$setOnInsert": {
                "created_at": now,
            },
        },
        upsert=True,
    )
    return True


def list_reference_topics(user_id: str, uri: str | None = None) -> list[dict[str, Any]]:
    """List all saved topics in the user's personal library."""
    db = get_db(uri)
    if db is None:
        return []
    col = db["reference_library"]
    clean_user = (user_id or "default_user").strip()
    cursor = col.find({"user_id": clean_user}).sort("updated_at", -1)
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results


def get_reference_topic(user_id: str, topic_name: str, uri: str | None = None) -> dict[str, Any] | None:
    """Fetch a specific saved topic."""
    db = get_db(uri)
    if db is None:
        return None
    col = db["reference_library"]
    clean_user = (user_id or "default_user").strip()
    doc = col.find_one({"user_id": clean_user, "topic_name": topic_name.strip()})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def delete_reference_topic(user_id: str, topic_name: str, uri: str | None = None) -> bool:
    """Delete a saved topic from the personal library."""
    db = get_db(uri)
    if db is None:
        return False
    col = db["reference_library"]
    clean_user = (user_id or "default_user").strip()
    res = col.delete_one({"user_id": clean_user, "topic_name": topic_name.strip()})
    return res.deleted_count > 0


# ==========================================================
# 2. Practice Attempts & Progress Tracking
# ==========================================================

def get_latest_attempt(user_id: str, topic_name: str, uri: str | None = None) -> dict[str, Any] | None:
    """Fetch the most recent attempt for this user and topic."""
    db = get_db(uri)
    if db is None:
        return None
    col = db["practice_attempts"]
    clean_user = (user_id or "default_user").strip()
    clean_topic = topic_name.strip()
    doc = col.find_one(
        {"user_id": clean_user, "topic_name": clean_topic},
        sort=[("attempt_number", -1)],
    )
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def save_practice_attempt(
    user_id: str,
    topic_name: str,
    expected_content: str,
    transcript: str,
    scores: dict[str, Any],
    tone: dict[str, Any],
    step_analysis: list[dict[str, Any]] | None = None,
    strengths: list[str] | None = None,
    improvements: list[str] | None = None,
    final_summary: str = "",
    sections: list[dict[str, Any]] | None = None,
    uri: str | None = None,
) -> dict[str, Any] | None:
    """Save practice attempt, calculate progress deltas compared to the previous attempt."""
    db = get_db(uri)
    if db is None:
        return None

    col = db["practice_attempts"]
    clean_user = (user_id or "default_user").strip()
    clean_topic = topic_name.strip()
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Fetch previous attempt for delta calculation
    prev = col.find_one(
        {"user_id": clean_user, "topic_name": clean_topic},
        sort=[("attempt_number", -1)],
    )

    attempt_number = (prev["attempt_number"] + 1) if prev else 1

    # 2. Calculate progress deltas
    delta: dict[str, Any] = {}
    if prev:
        prev_scores = prev.get("scores", {})
        delta = {
            "overall": int(scores.get("overall_score", 0)) - int(prev_scores.get("overall_score", 0)),
            "topic_relevance": int(scores.get("topic_relevance", 0)) - int(prev_scores.get("topic_relevance", 0)),
            "key_point_coverage": int(scores.get("key_point_coverage", 0)) - int(prev_scores.get("key_point_coverage", 0)),
            "content_accuracy": int(scores.get("content_accuracy", 0)) - int(prev_scores.get("content_accuracy", 0)),
            "clarity": int(scores.get("clarity", 0)) - int(prev_scores.get("clarity", 0)),
            "prev_attempt_number": prev.get("attempt_number", 1),
            "prev_date": prev.get("created_at"),
            "prev_overall": int(prev_scores.get("overall_score", 0)),
        }

    attempt_doc = {
        "user_id": clean_user,
        "topic_name": clean_topic,
        "attempt_number": attempt_number,
        "expected_content": expected_content.strip(),
        "transcript": transcript.strip(),
        "scores": scores,
        "tone": tone,
        "delta_vs_previous": delta,
        "step_analysis": step_analysis or [],
        "strengths": strengths or [],
        "improvements": improvements or [],
        "final_summary": final_summary,
        "sections": sections or [],
        "created_at": now,
    }

    res = col.insert_one(attempt_doc)
    attempt_doc["_id"] = str(res.inserted_id)
    return attempt_doc


def list_practice_attempts(user_id: str, topic_name: str | None = None, uri: str | None = None) -> list[dict[str, Any]]:
    """List historical attempts for a topic or user (for progress trend graphs)."""
    db = get_db(uri)
    if db is None:
        return []
    col = db["practice_attempts"]
    clean_user = (user_id or "default_user").strip()
    query: dict[str, Any] = {"user_id": clean_user}
    if topic_name:
        query["topic_name"] = topic_name.strip()

    cursor = col.find(query).sort("attempt_number", 1)
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results


# ==========================================================
# 3. Sectioned Speech Recording
# ==========================================================

def save_speech_section(
    user_id: str,
    topic_name: str,
    session_id: str,
    section_name: str,
    audio_path: str,
    transcript: str,
    tone: dict[str, Any] | None = None,
    order: int = 0,
    uri: str | None = None,
) -> bool:
    """Upsert a single speech section for a session (user + topic + session_id + section_name)."""
    db = get_db(uri)
    if db is None:
        return False

    col = db["speech_sections"]
    now = datetime.datetime.now(datetime.timezone.utc)
    clean_user = (user_id or "default_user").strip()
    clean_topic = topic_name.strip()
    clean_session = session_id.strip()
    clean_section = section_name.strip()

    col.update_one(
        {
            "user_id": clean_user,
            "topic_name": clean_topic,
            "session_id": clean_session,
            "section_name": clean_section,
        },
        {
            "$set": {
                "user_id": clean_user,
                "topic_name": clean_topic,
                "session_id": clean_session,
                "section_name": clean_section,
                "audio_path": audio_path,
                "transcript": transcript,
                "tone": tone or {},
                "order": order,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    return True


def list_speech_sections(
    user_id: str,
    topic_name: str,
    session_id: str,
    uri: str | None = None,
) -> list[dict[str, Any]]:
    """Return all sections for a session, sorted by order."""
    db = get_db(uri)
    if db is None:
        return []
    col = db["speech_sections"]
    clean_user = (user_id or "default_user").strip()
    cursor = col.find(
        {
            "user_id": clean_user,
            "topic_name": topic_name.strip(),
            "session_id": session_id.strip(),
        }
    ).sort("order", 1)
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results


def delete_speech_sections(
    user_id: str,
    topic_name: str,
    session_id: str,
    uri: str | None = None,
) -> bool:
    """Delete all sections for a session."""
    db = get_db(uri)
    if db is None:
        return False
    col = db["speech_sections"]
    clean_user = (user_id or "default_user").strip()
    res = col.delete_many(
        {
            "user_id": clean_user,
            "topic_name": topic_name.strip(),
            "session_id": session_id.strip(),
        }
    )
    return res.deleted_count > 0
