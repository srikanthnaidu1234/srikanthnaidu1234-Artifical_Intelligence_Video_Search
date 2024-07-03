import json
import logging
import os

import cv2
import psycopg2
import torch
import torch.nn.functional as F  # noqa: N812
from dotenv import load_dotenv
from torch import nn
from torchvision import transforms
from torchvision.datasets import CocoDetection

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create a logger object
logger = logging.getLogger(__name__)


# Define the autoencoder architecture
class Autoencoder(nn.Module):
    """Defines the autoencoder architecture using PyTorch nn.Module.

    The encoder consists of three convolutional layers followed by ReLU activations and
    max pooling layers.
    The decoder consists of three convolutional layers followed by ReLU activations and
    upsampling layers.

    Attributes:
    ----------
        encoder (nn.Sequential): The encoder part of the autoencoder.
        decoder (nn.Sequential): The decoder part of the autoencoder.

    """  # noqa: D406

    def __init__(self):  # noqa: ANN204, D107
        super(Autoencoder, self).__init__()  # noqa: UP008
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # noqa: ANN201, ANN001
        """Defines the forward pass of the autoencoder.

        Args:
        ----
            x (torch.Tensor): The input tensor of shape (batch_size, 1, 28, 28).

        Returns:
        -------
            torch.Tensor: The reconstructed output tensor shape (batch_size, 1, 28, 28).

        """  # noqa: D401
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  # noqa: RET504


def train_autoencoder_and_generate_embeddings(  # noqa: PLR0915
    coco_images_path: str,
    coco_annotations_path: str,
    num_epochs: int = 50,
    batch_size: int = 128,
) -> None:
    """Train an autoencoder on the COCO dataset and generate embeddings for
    detected objects in a Postgres database.

    The function filters the COCO dataset to only include the classes present
    in the Postgres table, trains the autoencoder on the filtered dataset, and
    then generates embeddings for the detected objects in the Postgres table using
    the trained encoder.

    Args:
    ----
        coco_images_path (str, optional): The path to the COCO dataset images. Defaults
        to "path/to/coco/images".
        coco_annotations_path (str, optional): The path to the COCO dataset annotations.
        Default to "path/to/coco/annotations/instances_train2017.json".
        num_epochs (int, optional): The number of training epochs for the autoencoder.
        Defaults to 50.
        batch_size (int, optional): The batch size for training the
        autoencoder. Defaults to 128.

    Returns:
    -------
        None

    """  # noqa: D205
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    cur = conn.cursor()
    try:
        cur.execute("""
            ALTER TABLE "detections"
            ADD COLUMN IF NOT EXISTS "embedding" JSONB;
        """)
        conn.commit()
    except psycopg2.errors.DuplicateColumn as e:
        logger.info(f"Column 'embedding' already exists in the 'detections' table: {e}")  # noqa: G004
    except Exception as e:  # noqa: BLE001
        logger.info(f"Error occurred: {e}")  # noqa: G004
        conn.rollback()
    finally:
        cur.close()
    cur = conn.cursor()
    # Check if the saved model exists
    model_path = "autoencoder_model.pth"
    if os.path.exists(model_path):  # noqa: PTH110
        autoencoder = Autoencoder()
        autoencoder.load_state_dict(torch.load(model_path))
        autoencoder.eval()
        logger.info("Loaded saved model.")
    else:
        # Train the autoencoder
        autoencoder = Autoencoder()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        # Fetch the unique class IDs from the Postgres table
        cur.execute('SELECT DISTINCT "detectedObjClass" FROM detections')
        class_ids = [row[0] for row in cur]
        # Load the COCO dataset and filter by the classes in the Postgres table
        coco_dataset = CocoDetection(
            root=coco_images_path,
            annFile=coco_annotations_path,
            transform=transforms.Compose(
                [
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                ]
            ),
            target_transform=lambda x: x in class_ids,
        )

        train_loader = torch.utils.data.DataLoader(
            coco_dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(num_epochs):
            for i, (images, _) in enumerate(train_loader):
                # Forward pass
                outputs = autoencoder(images)
                outputs = autoencoder(images)
                outputs = F.interpolate(
                    outputs, size=images.size()[2:], mode="bilinear", align_corners=True
                )

                loss = criterion(outputs, images)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{num_epochs}], \
                        Step [{i+1}/{len(coco_dataset)//batch_size}], \
                        Loss: {loss.item():.4f}"  # noqa: G004
                    )
        # Save the autoencoder model
        torch.save(autoencoder.state_dict(), "autoencoder_model.pth")

    cur.execute('SELECT DISTINCT "vidId" FROM "detections"')
    vid_ids = [row[0] for row in cur]
    logger.info(f"The video list is :{vid_ids}")  # noqa: G004
    # Add the "embedding" column with JSON type

    # Generate embeddings for detected objects
    for vid_id in set(vid_ids):
        query = """SELECT * FROM "detections" WHERE "vidId" = '{vid_id}';"""
        cur.execute(query)
        logger.info(cur.rowcount)
        # Open the video capture
        video_path = f"./download/{vid_id}.mp4"
        cap = cv2.VideoCapture(video_path)
        logger.info(f"The path for the video is {video_path}")  # noqa: G004
        saved_fps = 5  # The fps at which frame_idx was saved

        for row in cur:
            frame_idx = row[1]
            detection_idx = row[3]
            bbox = row[6]

            # Calculate the time of the frame
            frame_time = frame_idx / saved_fps

            # Set the video capture to the desired frame time
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)

            # Read the frame
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame {frame_idx} for video {vid_id}")  # noqa: G004
                continue

            # Crop the image based on the bounding box
            x, y, w, h = (int(value) for value in bbox)
            cropped_img = frame[y : y + h, x : x + w]

            # Resize the cropped image to match the input size of the encoder
            cropped_img = transforms.ToTensor()(cropped_img)
            cropped_img = transforms.Resize((28, 28))(cropped_img)
            cropped_img = cropped_img.unsqueeze(0)  # Add batch dimension

            # Generate the embedding using the encoder
            embedding = autoencoder.encoder(cropped_img).detach().numpy()

            # Update the Postgres table with the embedding
            embedding_json = json.dumps(embedding.tolist())
            cur.execute(
                'UPDATE "detections" SET "embedding" = %s WHERE "detectedObjId" = %s',
                (embedding_json, detection_idx),
            )

        cap.release()
        conn.commit()
    conn.close()
