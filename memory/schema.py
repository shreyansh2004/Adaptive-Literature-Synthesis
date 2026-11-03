"""
memory/schema.py

Defines the persistent memory schema for the research assistant.

Storage choice: SQLite.
Why: Day 17 requires "was this paper already read?" lookups and later,
cross-session queries like "what topics have I covered?". These are
relational, indexable lookups — SQLite handles them cleanly without
loading/scanning a whole JSON blob on every check. It's also stdlib
(no new dependency) and gives you a real interview talking point
("why SQLite over flat files") for free.

Four logical tables, matching the plan's Day 15 spec:
    - papers_read        : which arxiv papers have been fetched/read, and cached metadata
    - topics_covered      : which research topics have been searched/explored, and when
    - key_conclusions     : synthesized takeaways tied to a session/topic/paper
    - user_preferences    : simple key-value store for user preferences

A `sessions` table is included too — not in your original four, but needed
to group conclusions/topics by session for Day 17's "remember yesterday" test.
Feel free to drop it if you'd rather track sessions purely in-memory.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# DDL — run once at startup (store.py handles this via `init_db`)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    summary         TEXT
);

CREATE TABLE IF NOT EXISTS papers_read (
    arxiv_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    url             TEXT,
    abstract        TEXT,
    full_text_path  TEXT,          -- path to cached raw JSON in data/raw/, if fetched
    first_read_at   TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    session_id      TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS topics_covered (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic           TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    covered_at      TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS key_conclusions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    topic           TEXT,
    arxiv_id        TEXT,           -- nullable: conclusion may span multiple papers
    conclusion_text TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (arxiv_id) REFERENCES papers_read(arxiv_id)
);

CREATE TABLE IF NOT EXISTS user_preferences (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_topics_topic ON topics_covered(topic);
CREATE INDEX IF NOT EXISTS idx_conclusions_topic ON key_conclusions(topic);
CREATE UNIQUE INDEX IF NOT EXISTS idx_topic_session ON topics_covered(topic, session_id);
CREATE INDEX IF NOT EXISTS idx_papers_last_access ON papers_read(last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_papers_session ON papers_read(session_id);
CREATE INDEX IF NOT EXISTS idx_topics_session ON topics_covered(session_id);
CREATE INDEX IF NOT EXISTS idx_conclusions_session ON key_conclusions(session_id);
"""


# ---------------------------------------------------------------------------
# Lightweight dataclasses — convenient typed objects to pass around in
# planner.py / graph.py instead of raw sqlite3.Row objects.
# ---------------------------------------------------------------------------

@dataclass
class PaperRecord:
    arxiv_id: str
    title: str
    url: Optional[str] = None
    abstract: Optional[str] = None
    full_text_path: Optional[str] = None
    first_read_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: Optional[str] = None


@dataclass
class ConclusionRecord:
    session_id: str
    conclusion_text: str
    topic: Optional[str] = None
    arxiv_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MemorySnapshot:
    """
    What gets injected into the planner's context at the start of a session
    (Day 16: "inject relevant memory into the planner's context at session start").
    """
    recent_topics: list[str]
    recent_conclusions: list[str]
    known_paper_ids: set[str]
    preferences: dict[str, str]