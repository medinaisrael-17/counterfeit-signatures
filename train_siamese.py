# provides dataset, model, training + evaluation. 
# utilizes contrasive loss (Y=1 for same, 0 for different).
# includes a small synthetic dataset creation helper to test
# the pipeline quickly if no real data is present. 

# begin
import os
import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from glob import glob
from tqdm import tqdm
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, roc_curve, auc

# -------------------------
# Utilities / Dataset
# -------------------------

def default_image_transfrom(img_size=(155, 220)):
    # returns a torchvision transform (PIL in, tensor out)
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size[::-1]), # PIL size is (width, height)
        transforms.ToTensor(), # float in [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

class SignaturePairsDataset(Dataset): 
    """
    Expects root structures as:
    root/<person_id>/genuine/*.png
    root/<person_id>/forged/*.png (optional since might not have genuine forgeries)
    If forged/ is not available, negatives are created from other persons' genuine/. 
    """

    def __init__(self, root, transform=None, pairs_per_person=200):
        self.root = root
        self.person_dirs = sorted([p for p in glob(os.path.join(root, "*")) if os.path.isdir(p)])
        self.transform = transform or default_image_transform()
        self.pairs = []
        self._prepare_pairs(pairs_per_person)

    def _prepare_pairs(self, pairs_per_person):
        # gather file lists
        person_map = {}
        for p in self.person_dirs:
            pid = os.path.basename(p)
            g = glob(os.path.join(p, "genuine", "*"))
            f = glob(os.path.join(p, "forged", "*"))
            if len(g) < 1:
                continue
            person_map[pid] = {"genuine": g, "forged": f}
        pids = list(person_map.keys())
        # create pairs
        for pid in pids: 
            genuine = person_map[pid]["genuine"]
            forged = person_map[pid]["forged"]
            # positive pairs (genuine-genuine)
            for _ in range(int(pairs_per_person*0.4)):
                a, b = random.sample(genuine, 2) if len(genuine)>= 2 else (genuine[0], genuine[0])
                self.pairs.append((a, b, 1))
            # negative pairs using forgeries if available
            if len(forged) > 0:
                for _ in range(int(pairs_per_person*0.4)):
                    a = random.choice(genuine)
                    b = random.choice(forged)
                    self.pairs.append((a, b, 0))
            # negative pairs using other persons' genuine
            for _ in range(int(pairs_per_person*0.2)):
                other_pid = random.choice([x for x in pids if x != pid])
                b = random.choice(person_map[other_pid]["genuine"])
                a = random.choice(genuine)
                self.pairs.append((a, b, 0))
        random.shuffle(self.pairs)
