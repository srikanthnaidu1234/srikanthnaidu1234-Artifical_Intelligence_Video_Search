import logging  # noqa: INP001
import os

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from pytube import YouTube
from sqlalchemy.orm import Session
from youtube_transcript_api import YouTubeTranscriptApi

from .detecting_objects import detect_objects
from .embedding_model import train_autoencoder_and_generate_embeddings
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


class UpdateEmbeddingRequest(BaseModel):
    """Represents the request body schema for updating object embeddings."""

    video_id: str


class TrainingConfig(BaseModel):
    """Configuration model for autoencoder training and embedding generation.

    This Pydantic model defines the parameters required for training an autoencoder
    on the COCO dataset and generating embeddings for detected objects.

    Attributes:
    ----------
        coco_images_path (str): The file system path to the COCO dataset images.
            Defaults to "path/to/coco/images".
        coco_annotations_path (str): The file system path to the COCO dataset annotations file.
            Defaults to "path/to/coco/annotations/instances_train2017.json".
        num_epochs (int): The number of training epochs for the autoencoder.
            Defaults to 50.
        batch_size (int): The batch size to use during training.
            Defaults to 128.

    Example:
    -------
        ```python
        config = TrainingConfig(
            coco_images_path="/data/coco/images",
            coco_annotations_path="/data/coco/annotations/instances_train2017.json",
            num_epochs=100,
            batch_size=64
        )
        ```

    Note:
    ----
        - Ensure that the paths provided for `coco_images_path` and `coco_annotations_path`
        are valid and accessible.
        - Adjust `num_epochs` and `batch_size` based on your computational resources
        and dataset size for optimal training performance.

    """

    coco_images_path: str
    coco_annotations_path: str
    num_epochs: int = 50
    batch_size: int = 128


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
            video_id = yt.video_id
            logger.info(f"Downloading video: {yt.title}")  # noqa: G004
            video = yt.streams.get_highest_resolution()
            return video.download(self.output_dir, filename=video_id)
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
        # Save detections to the database
        for detection in detections:
            insert_detection(db, detection)

        return {"message": "object detection in frames is completed"}  # noqa: TRY300

    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to process frames: {e}")  # noqa: B904


@app.post("/train-and-generate")
async def train_and_generate(
    config: TrainingConfig, background_tasks: BackgroundTasks
) -> str:
    """
    Initiate the autoencoder training and embedding generation process.

    This endpoint triggers the training of an autoencoder on the COCO dataset
    and the generation of embeddings for detected objects. The process runs
    in the background to avoid blocking the API.

    Args:
    ----
        config (TrainingConfig): The configuration parameters for the training
            and embedding generation process. This includes paths to COCO dataset
            images and annotations, as well as training hyperparameters.
        background_tasks (BackgroundTasks): FastAPI's BackgroundTasks object
            used to run the process asynchronously.

    Returns:
    -------
        Dict[str, str]: A dictionary containing a message confirming that the
        process has been started in the background.

    Raises:
    ------
        HTTPException: If there's an error in initiating the background task.

    Example:
    -------
        ```
        POST /train-and-generate
        {
            "coco_images_path": "/path/to/coco/images",
            "coco_annotations_path": "/path/to/coco/annotations/instances_train2017.json",
            "num_epochs": 50,
            "batch_size": 128
        }
        ```

    Note:
    ----
        - This method does not wait for the training and embedding generation
        to complete. It immediately returns after initiating the background task.
        - The actual progress and completion of the task should be monitored
        through logs or a separate status endpoint.

    """  # noqa: D212
    # Run the function in the background
    background_tasks.add_task(
        train_autoencoder_and_generate_embeddings,
        coco_images_path=config.coco_images_path,
        coco_annotations_path=config.coco_annotations_path,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
    )
    return {"message": "Training and embedding generation started in the background"}
