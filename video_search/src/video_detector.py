import logging  # noqa: INP001
import os

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from pytube import YouTube
from sqlalchemy.orm import Session
from youtube_transcript_api import YouTubeTranscriptApi

from .detecting_objects import detect_objects
from .postgreSQL_db import get_db, insert_detection
from .video_indexing_pipeline import preprocess_video

app = FastAPI()

# Configure logging
# Set the logging level to INFO (DEBUG, INFO,WARNING, ERROR, CRITICAL)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create a logger object
logger = logging.getLogger(__name__)


class VideoInfo(BaseModel):
    """Represents the request body schema for video URL."""

    video_url: str


class DownloadResponse(BaseModel):
    """Represents the response schema for download endpoint."""

    video_path: str
    captions_file: str
    video_id: str


class YouTubeDownloader:
    """Utility class to handle YouTube video download and captions extraction."""

    def __init__(self, output_dir: str = "./download") -> None:
        """Initialize the YouTubeDownloader with the output directory.

        Args:
        ----
            output_dir (str, optional): Directory path where downloaded files
                will be saved. Defaults to "/app/downloads".

        """
        self.output_dir = output_dir

    def download_video(self, video_url: str) -> str:
        """Downloads a YouTube video.

        Args:
        ----
            video_url (str): The URL of the YouTube video.
            output_dir (str, optional): The directory to save the downloaded video.
            Defaults to "./".

        Returns:
        -------
            str: The path to the downloaded video file.

        Raises:
        ------
            HTTPException: If the video cannot be found or downloaded.

        """  # noqa: D401
        try:
            logger.info(video_url)
            yt = YouTube(video_url)
            logger.info(f"Downloading video: {yt.title}")  # noqa: G004
            video = yt.streams.get_highest_resolution()
            return video.download(self.output_dir)
        except KeyError:
            raise HTTPException(  # noqa: B904
                status_code=500, detail="Invalid response received from YouTube"
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Error downloading video: {e}")  # noqa: B904

    def download_captions(self, video_id: str) -> str:
        """Downloads captions for a YouTube video.

        Args:
        ----
            video_id (str): The ID of the YouTube video.
            output_dir (str, optional): The directory to save the captions file.
            Defaults to "./".

        Returns:
        -------
            str: The path to the captions file.

        Raises:
        ------
            HTTPException: If captions cannot be fetched or saved.

        """  # noqa: D401
        try:
            captions = YouTubeTranscriptApi.get_transcript(video_id)
            captions_file = os.path.join(self.output_dir, f"{video_id}_captions.txt")  # noqa: PTH118
            with open(captions_file, "w", encoding="utf-8") as f:  # noqa: PTH123
                for caption in captions:
                    f.write(
                        f"{caption['start']} --> {caption['start'] + caption['duration']}\n{caption['text']}\n\n"  # noqa: E501
                    )
            return captions_file  # noqa: TRY300
        except KeyError:
            raise HTTPException(  # noqa: B904
                status_code=500,
                detail="Invalid response received from YouTube Transcript API",
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(  # noqa: B904
                status_code=500, detail=f"Error downloading captions: {e}"
            )


downloader = YouTubeDownloader()


@app.post("/download", response_model=DownloadResponse)
async def download_video(video_info: VideoInfo) -> DownloadResponse:
    """Endpoint to download a YouTube video and its captions.

    Args:
    ----
        video_info (VideoInfo): The request body containing the video URL.

    Returns:
    -------
        DownloadResponse: The response containing the paths to the downloaded video and
        captions file.

    """
    try:
        video_url = video_info.video_url
        video_path = downloader.download_video(video_url)
        video_id = YouTube(video_url).video_id
        captions_file = downloader.download_captions(video_id)
        return DownloadResponse(
            video_path=video_path, captions_file=captions_file, video_id=video_id
        )
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(  # noqa: B904
            status_code=500, detail="Invalid response received from server"
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Server error: {e}")  # noqa: B904


@app.get("/preprocess")
async def preprocessing_video(video_path: str = Query(...)) -> str:
    """Endpoint to preprocess a video.

    Args:
    ----
        video_path (str): The path to the downloaded video file.

    """
    preprocess_video(video_path=video_path, output_dir="./processed_video_frames/")
    return {"message": "Video is downloaded in frames.npy"}


@app.get("/detect")
async def process_frames(
    video_id: str = Query(...),
    processed_video_frames: str = Query(...),
    db: Session = Depends(get_db),  # noqa: B008
) -> str:
    """Endpoint to process a video file located at the specified path,
    perform object detection on its frames,and return detected objects
    information in a structured format.

    Args:
    ----
        video_id (str): The ID of the video.
        processed_video_frames (str): The path to the frames.npy to process.
        db(session): the session with postgres db is done using get_db

    Returns:
    -------
        VideoInfo: A dictionary containing video information including detected objects.

    Raises:
    ------
        HTTPException(400): If there is an issue with the video file or processing.

    Note:
    ----
        This function assumes frames are already extracted and stored as frames.npy'.

    """  # noqa: D205
    try:
        logger.info("detecting of object is started")
        # Perform object detection on the video frames
        detections = detect_objects(video_id, processed_video_frames)
        logger.info(1000000000)
        # Save detections to the database
        for detection in detections:
            insert_detection(db, detection)

        return {"message": "object detection in frames is completed"}  # noqa: TRY300

    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to process frames: {e}")  # noqa: B904
