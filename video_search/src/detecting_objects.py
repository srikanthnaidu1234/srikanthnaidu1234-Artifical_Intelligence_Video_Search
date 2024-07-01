import logging  # noqa: INP001

import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Configure logging
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
    - List[Dict[str, any]]: A list of dictionaries, where each dictionary represents
    a detected object and contains keys:
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
        frames_raw = np.load(frames_file)
        frames = np.squeeze(frames_raw, axis=1)
        logger.info(f"Loaded {len(frames)} frames from {frames_file}")  # noqa: G004

        # Assuming you have the video ID and frame timestamps available
        vid_id = video_id
        frame_timestamps = [i * 0.2 for i in range(len(frames))]  # Example timestamps

        # Ensure frames are in [num_frames, height, width, channels] format
        # Convert the numpy array to a list of tensors
        frames = [torch.from_numpy(frame) for frame in frames]

        logger.info("Converted the numpy array to a list of torch tensors")

    except FileNotFoundError:
        raise FileNotFoundError(  # noqa: B904, TRY003
            f"File {frames_file} not found."  # noqa: EM102
        )  # Handle file not found error

    # Initialize the Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1")
    model.eval()

    detections = []

    with torch.no_grad():
        for idx, frame_tensor in enumerate(frames):
            # Ensure frame_tensor has correct shape [channels, height, width]
            output = model([frame_tensor.permute(2, 0, 1)])

            for detection_idx in range(len(output[0]["labels"])):
                label = output[0]["labels"][detection_idx].item()
                score = output[0]["scores"][detection_idx].item()
                bbox = output[0]["boxes"][detection_idx].tolist()

                detection = {
                    "vidId": vid_id,
                    "frameNum": idx,  # Use the index as the frame number
                    "timestamp": frame_timestamps[idx],
                    "detectedObjId": detection_idx,
                    "detectedObjClass": label,
                    "confidence": score,
                    "bbox": bbox,
                }

                detections.append(detection)

    return detections
