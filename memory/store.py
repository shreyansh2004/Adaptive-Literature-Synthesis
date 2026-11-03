"""
memory/store.py

MemoryStore: the read/write interface your planner and graph nodes use.
Wraps a local SQLite DB (default: memory/memory.db).

Day 15-17 usage pattern:
    store = MemoryStore()                          # Day 15: init
    store.start_session(session_id)
    snapshot = store.get_memory_snapshot()          # Day 16: inject into planner context
    ...
    if store.has_read_paper(arxiv_id):              # Day 17: skip re-fetch
        cached = store.get_paper(arxiv_id)
    else:
        # fetch normally via tools/paper_fetch.py, then:
        store.record_paper_read(arxiv_id, title, url, abstract, full_text_path)
    ...
    store.record_topic(topic)
    store.record_conclusion(conclusion_text, topic=topic, arxiv_id=arxiv_id)
    store.end_session(summary="...")
"""

import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from memory.schema import SCHEMA_SQL, PaperRecord, ConclusionRecord, MemorySnapshot

DEFAULT_DB_PATH = Path(__file__).parent / "memory.db"


class MemoryStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._current_session_id: Optional[str] = None
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        self.init_db()

    # -- setup -----------------------------------------------------------

    def init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- sessions ----------------------------------------------------------
    def _require_session(self):
        if self._current_session_id is None:
            raise RuntimeError(
                "No active session. Call start_session() first."
            )

    def start_session(self, session_id: Optional[str] = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        self._current_session_id = session_id
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, started_at) VALUES (?, ?)",
                (session_id, datetime.utcnow().isoformat()),
            )
        return session_id

    def end_session(self, summary: Optional[str] = None):
        self._require_session()
        with self._conn() as conn:
            if summary is not None:
                conn.execute(
                    "UPDATE sessions SET ended_at = ?, summary = ? WHERE session_id = ?",
                    (datetime.utcnow().isoformat(), summary, self._current_session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                    (datetime.utcnow().isoformat(), self._current_session_id),
                )
        self._current_session_id = None

    # -- papers_read ---------------------------------------------------------

    def has_read_paper(self, arxiv_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM papers_read WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
        return row is not None

    def get_paper(self, arxiv_id: str) -> Optional[PaperRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM papers_read WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
            if row is None:
                return None
            # touch last_accessed_at so "recently used" queries stay meaningful
            conn.execute(
                """UPDATE papers_read SET last_accessed_at = ? WHERE arxiv_id = ?""",
                (datetime.utcnow().isoformat(), arxiv_id),
            )
        return PaperRecord(**dict(row))

    def find_papers_by_title_keyword(self, keyword: str, limit: int = 5) -> list[PaperRecord]:
        """
        Case-insensitive whole-word match against papers_read.title.

        Uses LIKE only as a cheap prefilter, then enforces a word boundary in
        Python — a raw substring LIKE matched 'lora' inside 'expLORAtion',
        returning completely unrelated papers for "have you read the LoRA
        paper?". Results are ranked so the canonical paper (title starts with
        the keyword, e.g. "LoRA: Low-Rank Adaptation...") comes before variant
        titles, instead of whatever happened to be accessed most recently.
        """
        keyword = (keyword or "").strip()
        if not keyword:
            return []

        kw_l = keyword.lower()
        # Word boundary that treats hyphens as separators, so "Sync-LoRA"
        # still matches "lora" but "exploration" does not.
        boundary = re.compile(
            rf"(?<![a-zA-Z0-9]){re.escape(kw_l)}(?![a-zA-Z0-9])"
        )

        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM papers_read
                WHERE LOWER(title) LIKE ?
                ORDER BY last_accessed_at DESC
                LIMIT 100
                """,
                (f"%{kw_l}%",),
            ).fetchall()

        matches = [
            PaperRecord(**dict(row))
            for row in rows
            if boundary.search((row["title"] or "").lower())
        ]

        def rank(p: PaperRecord) -> int:
            title_l = (p.title or "").lower()
            if title_l.startswith(f"{kw_l}:"):
                return 0  # canonical: "LoRA: Low-Rank Adaptation..."
            if title_l.startswith(kw_l):
                return 1
            return 2

        matches.sort(key=rank)
        return matches[:limit]

    def list_known_papers(self, limit: int = 50) -> list[PaperRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM papers_read
                ORDER BY last_accessed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [PaperRecord(**dict(row)) for row in rows]

    def record_paper_read(
        self,
        arxiv_id: str,
        title: str,
        url: Optional[str] = None,
        abstract: Optional[str] = None,
        full_text_path: Optional[str] = None,
    ):
        self._require_session()
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO papers_read
                    (arxiv_id, title, url, abstract, full_text_path,
                     first_read_at, last_accessed_at, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    title = excluded.title,
                    url = excluded.url,
                    abstract = excluded.abstract,
                    full_text_path = excluded.full_text_path,
                    last_accessed_at = excluded.last_accessed_at,
                    session_id = excluded.session_id
                """,
                (arxiv_id, title, url, abstract, full_text_path,
                 now, now, self._current_session_id),
            )

    # -- topics_covered ------------------------------------------------------

    def record_topic(self, topic: str):
        """
        Upserts on (topic, session_id). A plain INSERT here would violate
        the UNIQUE(topic, session_id) constraint the moment the same topic
        is mentioned twice in one session (e.g. two follow-up turns both
        touching "LoRA fine-tuning"), which is completely normal usage, not
        an edge case. On a repeat mention we bump covered_at rather than
        silently ignoring it, so "recent topics" (ordered by covered_at
        DESC in get_recent_topics / get_memory_snapshot) actually reflects
        recency instead of freezing at the first-ever mention.
        """
        topic = topic.strip()  # only strip whitespace, preserve original case
        self._require_session()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO topics_covered (topic, session_id, covered_at)
                VALUES (?, ?, ?)
                ON CONFLICT(topic, session_id) DO UPDATE SET
                    covered_at = excluded.covered_at
                """,
                (topic, self._current_session_id, datetime.utcnow().isoformat()),
            )

    def get_recent_topics(self, limit: int = 10) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT topic FROM topics_covered ORDER BY covered_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r["topic"] for r in rows]

    def get_session_topics(self, session_id: str, limit: int = 10) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT topic FROM topics_covered WHERE session_id = ? ORDER BY covered_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [r["topic"] for r in rows]

    # -- key_conclusions -----------------------------------------------------

    def record_conclusion(
        self, conclusion_text: str, topic: Optional[str] = None, arxiv_id: Optional[str] = None
    ):
        self._require_session()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO key_conclusions (session_id, topic, arxiv_id, conclusion_text, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self._current_session_id, topic, arxiv_id, conclusion_text,
                 datetime.utcnow().isoformat()),
            )

    def get_recent_conclusions(self, limit: int = 10) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT conclusion_text FROM key_conclusions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r["conclusion_text"] for r in rows]

    def get_session_conclusions(self, session_id: str, limit: int = 10) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT conclusion_text FROM key_conclusions WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [r["conclusion_text"] for r in rows]

    # -- user_preferences ------------------------------------------------------

    def set_preference(self, key: str, value: str):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, datetime.utcnow().isoformat()),
            )

    def get_preferences(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM user_preferences").fetchall()
        return {r["key"]: r["value"] for r in rows}

    # -- the Day 16 hook: inject relevant memory into planner context --------

    def get_memory_snapshot(self, topic_limit: int = 5, conclusion_limit: int = 5) -> MemorySnapshot:
        with self._conn() as conn:
            topic_rows = conn.execute("""
                SELECT DISTINCT topic FROM topics_covered
                ORDER BY covered_at DESC LIMIT ?
                """, (topic_limit,)).fetchall()

            conclusion_rows = conn.execute(
                """
                SELECT conclusion_text FROM key_conclusions
                ORDER BY created_at DESC LIMIT ?
                """, (conclusion_limit,)).fetchall()

            paper_rows = conn.execute(
                "SELECT arxiv_id FROM papers_read"
            ).fetchall()

            preference_rows = conn.execute(
                "SELECT key, value FROM user_preferences"
            ).fetchall()

        return MemorySnapshot(
            recent_topics=[r["topic"] for r in topic_rows],
            recent_conclusions=[r["conclusion_text"] for r in conclusion_rows],
            known_paper_ids={r["arxiv_id"] for r in paper_rows},
            preferences={r["key"]: r["value"] for r in preference_rows},
        )