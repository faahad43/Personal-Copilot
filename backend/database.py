from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from models import Base
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database URL - uses PostgreSQL by default, falls back to SQLite for development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chat_database.db")

print(f"🔍 Using database: {DATABASE_URL.split('@')[0] + '@***' if '@' in DATABASE_URL else DATABASE_URL}")

# Create engine with appropriate settings for PostgreSQL vs SQLite
if DATABASE_URL.startswith("postgresql"):
    # PostgreSQL engine settings
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Test connections before using them
        pool_size=10,
        max_overflow=20,
        echo=False
    )
else:
    # SQLite engine settings
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database with tables"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized - tables created/verified")
    
    # Verify tables exist
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"✓ Tables in database: {tables}")
    
    # Test connection
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
            print("✓ Database connection verified")
    except Exception as e:
        print(f"⚠️  Database connection test failed: {e}")

@contextmanager
def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# For FastAPI dependency injection
def get_db_dep():
    """FastAPI dependency for database session"""
    db = SessionLocal()
    try:
        yield db
        db.commit()  # Ensure changes are persisted
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()