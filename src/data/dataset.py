"""
Custom PyTorch Datasets for Agricultural Pathology Classification.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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

    def _load_from_dataframe(self, df: pd.DataFrame, classes: Optional[Union[List[str], Dict[str, Any]]] = None) -> None:
        path_col = None
        for candidate in ["path", "image_path", "filepath", "file_path"]:
            if candidate in df.columns:
                path_col = candidate
                break

        if path_col is None or "label" not in df.columns:
            raise ValueError(
                f"Manifest DataFrame must contain an image path column ('path' or 'image_path') and 'label' column. "
                f"Found columns: {list(df.columns)}"
            )

        if classes is not None:
            if isinstance(classes, dict) and "classes" in classes:
                self.classes = sorted(list(classes["classes"]))
            elif isinstance(classes, dict):
                self.classes = sorted(list(classes.keys()))
            else:
                self.classes = sorted(list(classes))
        else:
            self.classes = sorted(df["label"].unique().tolist())

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for _, row in df.iterrows():
            lbl = str(row["label"])
            if lbl in self.class_to_idx:
                self.samples.append((str(row[path_col]), self.class_to_idx[lbl]))

    def _load_from_directory(self, root_dir: Path, classes: Optional[Union[List[str], Dict[str, Any]]] = None) -> None:
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if classes is not None:
            if isinstance(classes, dict) and "classes" in classes:
                self.classes = sorted(list(classes["classes"]))
            elif isinstance(classes, dict):
                self.classes = sorted(list(classes.keys()))
            else:
                self.classes = sorted(list(classes))
        else:
            subdirs = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
            self.classes = subdirs if subdirs else []

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Check if subdirectories matching classes exist
        has_subdirs = any((root_dir / cls_name).is_dir() for cls_name in self.classes)
        if has_subdirs:
            for cls_name in self.classes:
                cls_dir = root_dir / cls_name
                if not cls_dir.is_dir():
                    continue
                idx = self.class_to_idx[cls_name]
                for img_path in cls_dir.iterdir():
                    if img_path.suffix.lower() in valid_extensions:
                        self.samples.append((str(img_path), idx))
        else:
            # Fallback for flat directories (e.g. sample_real_leaves with files named Tomato___Early_blight_0.jpg)
            for img_path in root_dir.iterdir():
                if not img_path.is_file() or img_path.suffix.lower() not in valid_extensions:
                    continue
                fname = img_path.stem
                matched_cls = None
                for cls_name in self.classes:
                    if fname == cls_name or fname.startswith(f"{cls_name}_") or fname.startswith(f"{cls_name} ") or cls_name in fname:
                        matched_cls = cls_name
                        break
                if matched_cls is not None:
                    self.samples.append((str(img_path), self.class_to_idx[matched_cls]))

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
