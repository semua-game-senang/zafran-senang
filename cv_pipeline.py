"""
CV PIPELINE — Image Classification from Scratch
Dataset : CIFAR-10 (via torchvision — auto-downloads ~170 MB once)
Mirrors  : Any Kaggle image classification task
Tracks   : Manual patch extraction  →  2D CNN  →  TTA inference

Run: python cv_pipeline.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as T

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE  = 128
EPOCHS      = 15
LR          = 1e-3
IMG_SIZE    = 32    # CIFAR-10 images are 32×32
N_CLASSES   = 10
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print("=" * 60)
print("STAGE 0 : Config")
print("=" * 60)
print(f"  Device  : {DEVICE}")
print(f"  Classes : {CLASS_NAMES}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Data Loading + Transforms
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 1 : Loading CIFAR-10 + defining transforms")
print("=" * 60)
print("""
  WHY AUGMENTATION?
  The model should be invariant to:
    - horizontal flips (a car facing left = a car facing right)
    - small crops / translations
    - brightness / contrast shifts
  We apply these only at TRAIN time so the model doesn't overfit
  to exact pixel positions. At test time we use clean images.

  Normalization: bring each channel to mean≈0, std≈1.
  CIFAR-10 channel statistics (pre-computed from the dataset):
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]
""")

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

# Train: augment then normalize
train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),           # flip with 50% chance
    T.RandomCrop(32, padding=4),             # random crop (pad 4px on each side then crop)
    T.ColorJitter(brightness=0.2,            # random brightness ±20%
                  contrast=0.2,
                  saturation=0.2),
    T.ToTensor(),                            # HWC uint8 [0,255] → CHW float32 [0,1]
    T.Normalize(MEAN, STD),                  # (x - mean) / std per channel
])

# Test: only normalize, no augmentation
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

train_raw = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=train_transform)
test_raw  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=test_transform)

train_loader = DataLoader(train_raw, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_raw,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0)

print(f"  Train samples : {len(train_raw):,}")
print(f"  Test  samples : {len(test_raw):,}")
print(f"  Train batches : {len(train_loader)}")

# Show one batch
xb, yb = next(iter(train_loader))
print(f"\n  Batch X shape : {xb.shape}  → (batch, channels, height, width)")
print(f"  Batch y shape : {yb.shape}")
print(f"  Pixel range after normalize: [{xb.min():.2f}, {xb.max():.2f}]")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Patch Extraction (competition hint: sub-image cropping)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 2 : Patch extraction demo (for large-image tasks)")
print("=" * 60)
print("""
  The kisi-kisi says: "sub-image / patch cropping".
  This is relevant when images are LARGE (e.g. 512×512 or bigger).
  Strategy:
    1. Slide a window over the image → collect patches (e.g. 64×64)
    2. Classify each patch independently with the CNN
    3. Aggregate patch predictions → image-level prediction (majority vote / avg)

  For CIFAR-10 (32×32) this is not needed, but the function below
  is copy-paste ready for competition day.
""")

def extract_patches(img_chw: torch.Tensor, patch_size: int = 16, stride: int = 8):
    """
    img_chw : (C, H, W) float tensor
    Returns  : (N_patches, C, patch_size, patch_size)

    Example: 32×32 image, patch=16, stride=8
      → x positions: 0, 8, 16  (3 positions)
      → y positions: 0, 8, 16  (3 positions)
      → total = 9 patches
    """
    C, H, W = img_chw.shape
    patches  = []
    y_pos    = list(range(0, H - patch_size + 1, stride))
    x_pos    = list(range(0, W - patch_size + 1, stride))
    for y in y_pos:
        for x in x_pos:
            patch = img_chw[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return torch.stack(patches)   # (N, C, ph, pw)

sample_img = xb[0]   # (3, 32, 32)
patches    = extract_patches(sample_img, patch_size=16, stride=8)
print(f"  Image shape    : {sample_img.shape}")
print(f"  Patches shape  : {patches.shape}  → ({patches.shape[0]} patches of 16×16)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Model: 2D CNN from Scratch
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 3 : 2D CNN architecture")
print("=" * 60)
print("""
  BUILDING BLOCK — ConvBlock:
    Conv2d  : learnable filters that slide over the spatial dims
               kernel 3×3, same-padding → spatial size preserved
    BatchNorm : normalizes each channel's activations → stable training
    ReLU    : non-linearity (max(0, x))
    MaxPool : halves spatial resolution (2×2 → keeps strongest activation)

  STACK 4-5 ConvBlocks:
    Each block doubles the channels and halves the resolution.
    More channels = richer feature maps.
    Less spatial = bigger receptive field per activation.

    Input:  (B, 3, 32, 32)
    Block1: (B, 32, 16, 16)
    Block2: (B, 64, 8, 8)
    Block3: (B, 128, 4, 4)
    Block4: (B, 256, 2, 2)  [no pool on last block]
    AvgPool: (B, 256, 1, 1)
    Flatten → Linear(256, n_classes)
""")

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                              padding=kernel // 2,   # same-padding
                              bias=False)            # bias=False when using BN
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        # x → conv → BN → ReLU → (optional MaxPool)
        return self.pool(F.relu(self.bn(self.conv(x))))


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 10, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),          # (B,3,32,32) → (B,32,16,16)
            ConvBlock(32, 64),                   # → (B,64,8,8)
            ConvBlock(64, 128),                  # → (B,128,4,4)
            ConvBlock(128, 256, pool=False),     # → (B,256,4,4) — no pool
            nn.AdaptiveAvgPool2d((2, 2)),        # → (B,256,2,2) regardless of input size
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                        # (B,256,2,2) → (B,1024)
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),           # raw logits → CrossEntropyLoss
        )

    def forward(self, x):
        feat = self.features(x)                  # (B, 256, 2, 2)
        return self.classifier(feat)             # (B, n_classes)


model = SimpleCNN(n_classes=N_CLASSES).to(DEVICE)

print("  Model:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Trainable parameters: {total_params:,}")
print()

# Verify forward pass dimensions
with torch.no_grad():
    dummy  = torch.zeros(4, 3, 32, 32).to(DEVICE)
    out    = model(dummy)
    print(f"  Dummy forward: input {dummy.shape} → output {out.shape}  ✓")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — Training
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 4 : Training loop")
print("=" * 60)
print("""
  LOSS   : CrossEntropyLoss
            = -log(softmax(logit)[true_class])
            Penalizes confident wrong predictions heavily.

  OPTIMIZER : Adam (adaptive momentum — good default)
              lr=1e-3 with weight_decay=1e-4 (L2 regularization)

  SCHEDULER : CosineAnnealingLR
              LR starts at 1e-3, anneals smoothly to near 0.
              Better than step-decay; avoids abrupt LR drops.
""")

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0

print(f"  {'Epoch':>5} | {'LR':>8} | {'Loss':>8} | {'TrainAcc':>9} | {'ValAcc':>7}")
print("  " + "-" * 48)

for epoch in range(1, EPOCHS + 1):
    # ── Train ────────────────────────────────────────────────────────────
    model.train()
    total_loss = correct = total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb)                      # forward pass
        loss   = criterion(logits, yb)          # compute loss
        loss.backward()                         # backprop: compute ∂loss/∂params
        optimizer.step()                        # gradient descent step

        total_loss += loss.item() * len(yb)
        preds       = logits.argmax(dim=1)      # predicted class = highest logit
        correct    += (preds == yb).sum().item()
        total      += len(yb)

    train_loss = total_loss / total
    train_acc  = correct / total

    # ── Validate ──────────────────────────────────────────────────────────
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():                       # no gradients needed at test time
        for xb, yb in test_loader:
            xb, yb   = xb.to(DEVICE), yb.to(DEVICE)
            preds     = model(xb).argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total   += len(yb)
    val_acc = val_correct / val_total

    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()                            # update LR after each epoch

    flag = " ← best" if val_acc > best_val_acc else ""
    print(f"  {epoch:>5} | {current_lr:>8.5f} | {train_loss:>8.4f} | {train_acc:>9.4f} | {val_acc:>7.4f}{flag}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_cnn.pth')

print(f"\n  Best validation accuracy: {best_val_acc:.4f}")
print("  Model saved to best_cnn.pth")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Inference + Test-Time Augmentation (TTA)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STAGE 5 : Inference with Test-Time Augmentation (TTA)")
print("=" * 60)
print("""
  TTA = run the test image through the model multiple times,
  each time with a DIFFERENT random augmentation, then average
  the probability outputs.

  Why it works: each augmented view gives a slightly different
  probability vector. Averaging reduces variance → better calibration.

  Cost: N_TTA × inference time. Usually N_TTA=5 is enough.
  Gain: typically +0.5–2% accuracy for free.
""")

# Reload best weights
model.load_state_dict(torch.load('best_cnn.pth', map_location=DEVICE))
model.eval()

# TTA transform: same as train (random flips/crops each call)
tta_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

def predict_tta(model, dataset, n_tta: int = 5, batch_size: int = 256):
    """
    Runs TTA: creates n_tta different augmented views, averages softmax probs.
    Returns (N_samples, N_classes) probability matrix.
    """
    all_probs = []

    # We need the raw dataset (PIL images) to apply multiple transforms
    tta_ds     = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=False, transform=tta_transform)
    tta_loader = DataLoader(tta_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    for run in range(n_tta):
        run_probs = []
        with torch.no_grad():
            for xb, _ in tta_loader:
                xb    = xb.to(DEVICE)
                probs = torch.softmax(model(xb), dim=1).cpu().numpy()
                run_probs.append(probs)
        all_probs.append(np.vstack(run_probs))
        print(f"    TTA run {run+1}/{n_tta} done")

    return np.mean(all_probs, axis=0)   # average over all TTA runs

print("  Running TTA (5 passes)...")
tta_probs  = predict_tta(model, test_raw, n_tta=5)
tta_preds  = tta_probs.argmax(axis=1)
true_labels = np.array(test_raw.targets)

tta_acc = (tta_preds == true_labels).mean()
print(f"\n  TTA Accuracy : {tta_acc:.4f}")
print(f"  Best val acc : {best_val_acc:.4f}")
print(f"  TTA boost    : {tta_acc - best_val_acc:+.4f}")

# Per-class breakdown
print("\n  Per-class accuracy:")
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    mask     = true_labels == cls_idx
    cls_acc  = (tta_preds[mask] == true_labels[mask]).mean()
    bar      = '█' * int(cls_acc * 20)
    print(f"    {cls_name:>12} : {bar:<20} {cls_acc:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — Quick Submission Template
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STAGE 6 : Submission (mock — swap paths on Kaggle day)")
print("=" * 60)
print("""
  On Kaggle:
    test_df = pd.read_csv('/kaggle/input/.../sample_submission.csv')
    test_df['label'] = tta_preds          # or le.inverse_transform(tta_preds)
    assert test_df['label'].isnull().sum() == 0
    assert len(test_df) == len(test images)
    test_df.to_csv('submission.csv', index=False)
""")

print("  Mock submission head:")
import pandas as pd
mock_sub = pd.DataFrame({'id': range(len(tta_preds)),
                          'label': [CLASS_NAMES[p] for p in tta_preds]})
print(mock_sub.head(10).to_string(index=False))
print()

print("=" * 60)
print("  SUMMARY — Key numbers")
print("=" * 60)
print(f"  Single-pass accuracy : {best_val_acc:.4f}")
print(f"  TTA accuracy (5x)    : {tta_acc:.4f}")
print("""
  Key competition decisions:
    1. Always normalize images (pixel / 255, then (x-mean)/std).
    2. RandomCrop + HorizontalFlip = cheapest augmentation.
    3. AdaptiveAvgPool → model handles any input resolution.
    4. TTA = free boost, add it in the last 30 min.
    5. If images are large: extract patches → classify → majority vote.
""")
