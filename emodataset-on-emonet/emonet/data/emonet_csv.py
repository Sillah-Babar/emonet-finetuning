from pathlib import Path
import random

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


#  dataset 
class EmoNetCSV(Dataset):
    """
    CSV columns expected: pth,label,valence,arousal
      - pth     : image file name or relative path
      - label   : string expression (optional use)
      - valence : float in [-1,1]
      - arousal : float in [-1,1]
    """
    def __init__(self, csv_path, root, size=256, use_expr=True, label2id=None, augment=False):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.size = size
        self.use_expr = use_expr
        self.label2id = label2id or {}
        # self.augment = augment

        required = ["pth", "valence", "arousal"]
        for c in required:
            assert c in self.df.columns, f"CSV must contain column '{c}'"
        if self.use_expr:
            assert "label" in self.df.columns, "CSV must contain 'label' when use_expr=True"

    def __len__(self): return len(self.df)

    def _augment(self, img):
        # mild & expression-safe aug
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.3:
            h, w = img.shape[:2]
            ang = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if random.random() < 0.3:
            alpha = 1.0 + random.uniform(-0.1, 0.1) # contrast
            beta = random.uniform(-10, 10)          # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img

    def __getitem__(self, idx):
        # Load & preprocess image
        row = self.df.iloc[idx]
        img_path = self.root / str(row["pth"])
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(img_path)
        bgr = cv2.resize(bgr, (self.size, self.size))
        # if self.augment: bgr = self._augment(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        v = float(row["valence"]); a = float(row["arousal"])
        y = {"valence": torch.tensor(v, dtype=torch.float32),
             "arousal": torch.tensor(a, dtype=torch.float32)}
        if self.use_expr:
            lab = normalize_label(row["label"])
            y["expr"] = torch.tensor(self.label2id[lab], dtype=torch.long)

        return x, y


# Normalize incoming label strings to lowercase without surrounding spaces
def normalize_label(s: str) -> str:
    return str(s).strip().lower()

