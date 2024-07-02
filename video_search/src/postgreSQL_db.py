import os
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import JSON, Column, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

# DATABASE_URL = "postgresql://postgres:pwd@localhost/dbname"  # noqa: ERA001

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Detection(Base):
    """SQLAlchemy model for the detections table.

    Attributes
    ----------
        id (int): Primary key for the detection.
        vidId (str): ID of the video.
        frameNum (int): Frame number of the detection.
        timestamp (float): Timestamp of the detection.
        detectedObjId (int): ID of the detected object.
        detectedObjClass (int): Class of the detected object.
        confidence (float): Confidence score of the detection.
        bbox (JSON): Bounding box coordinates of the detection.

    """

    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    vidId = Column(String)  # noqa: N815
    frameNum = Column(Integer)  # noqa: N815
    timestamp = Column(Float)
    detectedObjId = Column(Integer)  # noqa: N815
    detectedObjClass = Column(Integer)  # noqa: N815
    confidence = Column(Float)
    bbox = Column(JSON)


Base.metadata.create_all(bind=engine)


def get_db() -> any:
    """Creates a database session and handles its lifecycle.

    Yields:
    ------
        Session: A SQLAlchemy database session.

    Note:
    ----
        This function is typically used as a dependency in FastAPI.
        It ensures that the database session is properly closed after use.

    """  # noqa: D401
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def insert_detection(db: Session, detection: dict[str, Any]) -> Detection:
    """Insert a new detection record into the database.

    Args:
    ----
        db (Session): The database session.
        detection (Dict[str, Any]): A dictionary containing detection data.

    Returns:
    -------
        Detection: The inserted Detection object.

    Note:
    ----
        This function creates a new Detection object from the provided dictionary,
        adds it to the database, commits the transaction, and refreshes the object
        to ensure it contains any database-generated values.

    """
    db_detection = Detection(**detection)
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    return db_detection
