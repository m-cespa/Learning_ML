import numpy as np
from typing import List
import os
from pathlib import Path
from PIL import Image
import random

class DataLoader:
    """
    Generates object to load into network.
    """
    def __init__(self, path: str):
        if Path(path).exists():
            self.path = Path(path)
        else:
            raise ValueError('Path to data does not exist.')
        
        self.labels_images = []

        self.load_images()
        
    def load_images(self, shuffle: bool) -> None:
        """
        For image type data.
        Creates input and label attributes for Loader object.
        """
        if self.path.is_dir():
            for label_dir in self.path.iterdir():
                label = label_dir.name
                images = []

                image_count = 0
                for image_file in label_dir.iterdir():
                    if image_file.is_file() and image_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                        image_tensor = np.array(Image.open(image_file))
                        images.append(image_tensor)
                        image_count += 1

                self.labels_images.append((label, image_count, images))
        else:
            raise ValueError('Images sub-directory absent.')
        
        if shuffle:
            random.shuffle(self.labels_images)
        
    def batchify(self, batch_dim: int) -> List[np.ndarray, np.ndarray]:
        """
        Randomly selects tensors from self.labels_images.
        Concatenates tensors over the batch dimension.
        
        batch_images shape: (batch_dim, height, width, channels)
        batch_labels shape: (batch_dim,)
        """
        batch_images = []
        batch_labels = []

        flat_images = []
        flat_labels = []

        for label_idx, (label, image_count, images) in enumerate(self.labels_images):
            flat_images.extend(images)
            flat_labels.extend([label_idx] * image_count)

        indices = random.sample(range(len(flat_images)), batch_dim)

        for idx in indices:
            batch_images.append(flat_images[idx])
            batch_labels.append(flat_labels[idx])

        batch_images = np.stack(batch_images)
        batch_labels = np.array(batch_labels)

        return [batch_labels, batch_images]
