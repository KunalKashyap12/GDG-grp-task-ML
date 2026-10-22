"""
Custom PyTorch Datasets for Agricultural Pathology Classification.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class CropDiseaseDataset(Dataset):
    """
    Production-grade Dataset supporting either directory-based image hierarchies
    (e.g., 'processed/train/Tomato___Early_blight/img.jpg') or CSV manifest tables.
    """

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        classes: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        if isinstance(data_source, pd.DataFrame):
            self._load_from_dataframe(data_source, classes)
        elif Path(data_source).is_file() and str(data_source).endswith(".csv"):
            df = pd.read_csv(data_source)
            self._load_from_dataframe(df, classes)
        elif Path(data_source).is_dir():
            self._load_from_directory(Path(data_source), classes)
        else:
            raise ValueError(f"Invalid data_source: {data_source}. Expected directory or CSV manifest.")

    def _load_from_dataframe(self, df: pd.DataFrame, classes: Optional[List[str]] = None) -> None:
        if "image_path" not in df.columns or "label" not in df.columns:
            raise ValueError("Manifest DataFrame must contain 'image_path' and 'label' columns.")

        if classes is not None:
            self.classes = sorted(list(classes))
        else:
            self.classes = sorted(df["label"].unique().tolist())

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for _, row in df.iterrows():
            lbl = str(row["label"])
            if lbl in self.class_to_idx:
                self.samples.append((str(row["image_path"]), self.class_to_idx[lbl]))

    def _load_from_directory(self, root_dir: Path, classes: Optional[List[str]] = None) -> None:
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if classes is not None:
            self.classes = sorted(list(classes))
        else:
            self.classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = root_dir / cls_name
            if not cls_dir.is_dir():
                continue
            idx = self.class_to_idx[cls_name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Loads image safely and applies transformations.

        Returns:
            image_tensor: Shape [3, Height, Width]
            label: integer class index
        """
        img_path, label = self.samples[index]

        try:
            with Image.open(img_path) as img:
                image_rgb = img.convert("RGB")
        except Exception:
            # Fallback zero tensor for unreadable or corrupt file
            return torch.zeros((3, 224, 224), dtype=torch.float32), label

        if self.transform is not None:
            image_tensor = self.transform(image_rgb)
        else:
            image_tensor = torch.from_numpy(np.array(image_rgb)).permute(2, 0, 1).float() / 255.0

        return image_tensor, label
