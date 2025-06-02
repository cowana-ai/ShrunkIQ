import numpy as np
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# Global singleton instance
_lpips_instance = None

def compute_image_lpips(
    image1: np.ndarray | Image.Image,
    image2: np.ndarray | Image.Image,
    net_type: str = 'vgg',
) -> float:
    global _lpips_instance
    if _lpips_instance is None:
        _lpips_instance = LPIPS(net_type)
    return _lpips_instance(image1, image2)


class LPIPS:
    def __init__(self, net_type: str = 'vgg'):
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type)
        # Transform pipeline: PIL → Tensor in [-1, 1] → Resized
        self.transform = Compose([
            Resize((224, 224)),  # Resize for consistency
            ToTensor(),  # Converts to [0, 1]
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Rescales to [-1, 1]
        ])

    def __call__(self, image1: np.ndarray | Image.Image, image2: np.ndarray | Image.Image) -> float:
        image1 = self.transform(image1).unsqueeze(0)
        image2 = self.transform(image2).unsqueeze(0)
        return self.lpips(image1, image2).item()
