import logging

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Configure logging
# Set the logging level to INFO (DEBUG, INFO,WARNING, ERROR, CRITICAL)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create a logger object
logger = logging.getLogger(__name__)


def detect_objects(video_id: str, frames_file: str) -> list[dict[str, any]]:  # noqa: D417
    """Perform object detection on frames stored in a numpy file.

    Args:
    ----
    - video_id (str): The unique identifier of the video.
    - frames_file (str): Path to the numpy file containing frames.

    Returns:
    -------
    - List[Dict[str, any]]: A list of dictionaries, where each dictionary represents\
    a detected object and contains the following keys:
        - 'vidId': The unique identifier of the video.
        - 'frameNum': The number of the video frame.
        - 'timestamp': The timestamp of the video frame.
        - 'detectedObjId': Index of the detected object.
        - 'detectedObjClass': Class label of the detected object.
        - 'confidence': Confidence score of the detection.
        - 'bbox': Bounding box coordinates of the detection.

    Raises:
    ------
    - FileNotFoundError: If the frames_file does not exist.

    Note:
    ----
    - This function assumes a pretrained Faster R-CNN model from torchvision.

    """
    try:
        # Load frames and metadata from npy file
        frames = np.load(frames_file)
        logger.info(f"Loaded {len(frames)} frames from {frames_file}")  # noqa: G004

        # Assuming you have the video ID and frame timestamps available
        vid_id = video_id
        frame_timestamps = [i * 0.2 for i in range(len(frames))]  # Example timestamps

    except FileNotFoundError:
        raise FileNotFoundError(f"File {frames_file} not found.")  # noqa: B904, TRY003, EM102

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    logger.info("The model is pretrained by faster-rcnn using MS COCO classes")
    model.eval()

    detections = []
    logger.info("checking the frames for the object similar to MS coco clases")
    for frame_num, frame in enumerate(frames):
        pil_img = Image.fromarray(frame)
        img_tensor = transforms.ToTensor()(pil_img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model([img_tensor])  # Pass the tensor as a list

        for idx in range(len(output[0]["labels"])):
            label = output[0]["labels"][idx].item()
            score = output[0]["scores"][idx].item()
            bbox = output[0]["boxes"][idx].tolist()

            detection = {
                "vidId": vid_id,
                "frameNum": frame_num,
                "timestamp": frame_timestamps[frame_num],
                "detectedObjId": idx,
                "detectedObjClass": label,
                "confidence": score,
                "bbox": bbox,
            }

            detections.append(detection)

    return detections
