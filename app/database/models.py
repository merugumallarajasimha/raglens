"""SQLAlchemy ORM models for papers, chunks, and authors."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import ForeignKey, Table, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import relationship

from app.database.postgres import Base

paper_authors = Table(
    "paper_authors",
    Base.metadata,
    Column("paper_id", ForeignKey("papers.id"), primary_key=True),
    Column("author_id", ForeignKey("authors.id"), primary_key=True),
)


class Paper(Base):
    __tablename__ = "papers"

    id = Column(String, primary_key=True, index=True)  # paper_id, e.g. arxiv ID
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    year = Column(Integer)
    source = Column(String(50))
    url = Column(String(500))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    authors = relationship("Author", secondary=paper_authors, back_populates="papers")
    chunks = relationship("Chunk", back_populates="paper", cascade="all, delete-orphan")
    sections = relationship("Section", back_populates="paper", cascade="all, delete-orphan")


class Author(Base):
    __tablename__ = "authors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, unique=True)

    papers = relationship("Paper", secondary=paper_authors, back_populates="authors")


class Section(Base):
    __tablename__ = "sections"

    id = Column(String, primary_key=True, index=True)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    heading = Column(Text, nullable=False)
    level = Column(Integer, default=1)
    order_index = Column(Integer, default=0)

    paper = relationship("Paper", back_populates="sections")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, index=True)  # chunk_id
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False, index=True)
    section = Column(String(200))
    subsection = Column(String(200))
    page_start = Column(Integer)
    page_end = Column(Integer)
    text = Column(Text, nullable=False)
    token_count = Column(Integer)

    paper = relationship("Paper", back_populates="chunks")
