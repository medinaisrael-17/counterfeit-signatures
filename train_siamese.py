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

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        a_path, b_path, label = self.pairs[idx]
        a = Image.open(a_path).convert("RGB")
        b = Image.open(b_path).convert("RGB")
        if self.transform:
            a = self.transform(a)
            b = self.transform(b)
        return a, b, torch.tensor(label, dtype=torch.float32)
    
# -------------------------
# Model: Siamese network
# -------------------------
class SmallCNN(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1), # in channels 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # /2
            nn.Conv2d(35, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # /4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # /8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.embedding_net = SmallCNN(embedding_dim=embedding_dim)

    def forward(self, x1, x2):
        e1 = self.embedding_net(x1)
        e2 = self.embedding_net(x2)
        return e1, e2
    
# Contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label: 1 for similar, 0 for dissimilar
        distances = F.pairwise_distance(output1, output2)
        loss_similar = label * torch.pow(distances, 2)
        loss_dissimilar = (1 - label) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        loss = 0.5 * (loss_similar + loss_dissimilar)
        return loss.mean()
    

# -------------------------
# Training & Evaluation
# -------------------------
def evaluate(model, dataloader, device):
    model.eval()
    labels = []
    distances = []
    with torch.no_grad():
        for a, b, y in dataloader:
            a = a.to(device); b = b.to(device)
            e1, e2 = model(a, b)
            d = F.pairwise_distance(e1, e2).detach().cpu().numpy()
            distances.extend(d.tolist())
            labels.extend(y.numpy().tolist()) # ?
    # compute AUC
    auc_score = roc_auc_score(labels, [-x for x in distances]) # invert distances -> score
    fpr, tpr, thresholds = roc_curve(labels, [-x for x in distances])
    return auc_score, fpr, tpr, thresholds, distances, labels

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = default_image_transfrom(img_size=(args.height, args.width))
    train_ds = SignaturePairsDataset(args.train_dir, transform=transform, pairs_per_person=args.pairs_per_person)
    val_ds = SignaturePairsDataset(args.val_dir, transform=transform, pairs_per_person=max(50, args.pairs_per_person//4))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SiameseNetwork(embedding_dim=args.embedding_dim).to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_auc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        for a, b, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            a = a.to(device); b = b.to(device); y = y.to(device)
            optimizer.zero_grad()
            e1, e2 = model(a, b)
            loss = criterion(e1, e2, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * a.size(0)
        scheduler.step()
        avg_loss = running_loss / len(train_loader.dataset)
        auc_score, _, _, _, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, val_auc={auc_score:.4f}")
        # Save best
        if auc_score > best_auc:
            best_auc = auc_score
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved best model (AUC {best_auc:.4f}) -> {args.checkpoint}")

    print("Training finished. Best val AUC:", best_auc)


# -------------------------
# Small synthetic dataset builder (for testing)
# -------------------------
def make_synthetic_dateset(root, persons=10, genuines_per=6, img_size=(220,155)):
    # generates synthetic "signature-like" images using PIL text to simulate handwriting
    os.makedirs(root, exist_ok=True)
    import string
    from PIL import ImageDraw, ImageFont
    try: 
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
    for i in range(persons):
        pid = f"person_{i:03d}"
        gdir = os.path.join(root, pid, "genuine")
        fdir = os.path.join(root, pid, "forged")
        os.makedirs(gdir, exist_ok=True)
        os.makedirs(fdir, exist_ok=True)
        name = "User" + str(i)
        for j in range(genuines_per):
            img = Image.new("RGB", img_size, "white")
            draw = ImageDraw.Draw(img)
            # draw name with small jitter to simulate variations
            for _ in range(random.randint(1, 3)):
                x1,y1 = random.randint(0,img_size[0]//2), random.randint(0,img_size[1])
                x2,y2 = random.randint(img_size[0]//2,img_size[0]), random.randint(0, img_size[1])
                draw.line((x1,y1,x2,y2), fill=(0,0,0), width=random.randint(1,3))
            fname = os.path.join(gdir, f"g_{j}.png")
            img.save(fname)
        # forgeries: use another person's name or distort
        for j in range(max(3, genuines_per//2)):
            img = Image.new("RGB", img_size, "white")
            draw = ImageDraw.Draw(img)
            other = "Fake" + str(random.randint(0, persons*2))
            draw.text((random.randint(5,20), random.randint(30,70)), other, fill=(0,0,0), font=font)
            fname = os.path.join(fdir, f"f_{j}.png")
            img.save(fname)


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--checkpoint", type=str, default="siamese_best.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--pairs_per_person", type=int, default=300)
    parser.add_argument("--width", type=int, default=155)
    parser.add_argument("--height", type=int, default=220)
    parser.add_argument("--make_synth", action="store_true", help="Make Synthetic dataset for quick test")

    args = parser.parse_args()

    if args.make_synth:
        print("Making synthetic train/val datasets...")
        make_synthetic_dateset("data/train", persons=40, genuines_per=8, img_size=(args.width, args.height))
        make_synthetic_dateset("data/val", persons=12, genuines_per=6, img_size=(args.width, args.height))
        print("Done synthetci data.")
    train(args)
