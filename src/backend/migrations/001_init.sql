-- PaperLens metadata schema (PostgreSQL / Supabase)
-- Binary PDFs and images remain in object storage (S3/local), not in these tables.
-- Applied automatically via SQLAlchemy Base.metadata.create_all for local SQLite.
-- For Supabase, run create_all against DATABASE_URL or adapt to hosted migrations.

-- papers, paper_sections, paper_elements, visual_elements, paper_chunks, jobs
-- See src/backend/app/db/models.py for the canonical ORM definition.
