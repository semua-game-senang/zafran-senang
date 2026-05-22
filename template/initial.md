

Let me first read the skill files and scrape some relevant resources before diving in.Now let me scrape some relevant Kaggle solutions and resources:Now I have enough context. Let me build the comprehensive cheatsheet document:Now let me create the comprehensive cheatsheet document:Here's the full overview before you dive into the doc, since you asked for that first:

---

## Overview: The Mental Model for This Competition

The competition format — 2 problems in 5 hours, Kaggle only, no pretrained models, judged on private leaderboard — is essentially **"get something on the board fast, then iterate."** The reference to the AICC leaderboard you linked is telling: those competitions reward a **single clean pipeline** over complex ensembles. Here's the landscape:

**Tabular** is your safest track. LightGBM is the most popular GBDT library among winners, and for tabular competitions, gradient boosted decision trees (XGBoost, LightGBM, CatBoost) still dominate — and feature quality matters more than model parameters. The recipe is: LightGBM with 5-fold StratifiedKFold, early stopping, and a light XGBoost as a second model to blend. You can have a scored submission in 15 minutes.

**NLP** has two sub-types per the kisi-kisi: sentence classification and token-level classification (NER/POS). The constraint "no pretrained model, rakit sendiri RNN/LSTM/Transformer" means you're building embedding → BiLSTM → head from scratch. The baseline approach is: tokenize each sentence, encode tokens as indices, pass through an embedding layer, then a two-stacked LSTM whose last hidden state feeds a linear classifier. However — **always start with TF-IDF + LogReg first.** For large datasets or cases where the number of tokens is small, experimenting with traditional vectorization methods combined with machine learning is a very beneficial first step. It often beats LSTM on small data within the time constraint.

For **token-level classification** (NER), the gold standard from-scratch architecture is **BiLSTM-CRF**. The Bi-LSTM CRF is essential for strong NER performance — the CRF imposes sequential constraints, while the LSTM provides the features. A simpler BiLSTM tagger (without CRF) is typically sufficient for POS tagging. If time is short, skip the CRF and just use BiLSTM → linear → softmax per token; you'll lose a few F1 points but save an hour.

**CV** is the hardest under constraints. The kisi-kisi explicitly mentions "patch/sub-image cropping" and standard OpenCV processing — this is a hint that images may be large and the task requires sliding-window patch extraction before feeding a CNN. Build Conv→BN→ReLU→MaxPool blocks stacked 4-5 deep with an adaptive average pool at the end. No fancy architecture needed. Test-Time Augmentation (TTA) is a free ~1-2% accuracy boost at inference time.

**Team split** matters hugely at 5 hours. Three people should own one track each, share a Kaggle team, and merge notebooks only in the last 30 minutes for blending.

The cheatsheet above has all the copy-paste-ready code for every scenario. Save it offline and load it into a Kaggle notebook as your starting cell. Good luck tomorrow! 🎯# 🏆 Gemastik / Satria Data — Internal Selection Cheatsheet


**Team of 3 | 5 hours | Offline + Kaggle only | No AI coding tools | No pretrained models**
> Read first. Code second. Submit fast.

---

## 🗺️ OVERVIEW: The Two-Problem Strategy

You get **2 problems** drawn from {Tabular, NLP, CV}. Here's the meta-strategy:

| Track | "Key Pipeline" one-liner | Fastest path to score |
|---|---|---|
| **Tabular** | LightGBM + feature engineering + CV | XGBoost/LGBM baseline in 15 min, iterate features |
| **NLP — Classification** | TF-IDF + LogReg OR LSTM (Embedding→BiLSTM→Linear) | TF-IDF baseline in 10 min, LSTM if time allows |
| **NLP — Token (NER/POS)** | BiLSTM-CRF | Char+Word embed → BiLSTM → softmax (skip CRF if time-pressed) |
| **CV — Classification** | Patch crop → 2D CNN (Conv→BN→ReLU→Pool stack) | Small CNN + augmentation in 30 min |

**Team split suggestion (3 people):**
- Person A → EDA + feature engineering + tabular model
- Person B → NLP pipeline
- Person C → CV pipeline
- Everyone submits to Kaggle independently, merge best notebooks at end.

---

## 📦 PART 1 — TABULAR (Classification & Regression)

### 1.1 The Universal Template

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

# ── LOAD ──────────────────────────────────────────────────────────────
train = pd.read_csv('/kaggle/input/.../train.csv')
test  = pd.read_csv('/kaggle/input/.../test.csv')
sub   = pd.read_csv('/kaggle/input/.../sample_submission.csv')

TARGET = 'target'
ID_COL = 'id'

# ── EDA (2 minutes) ────────────────────────────────────────────────────
print(train.shape, test.shape)
print(train[TARGET].value_counts())
print(train.isnull().sum().sort_values(ascending=False).head(20))
print(train.dtypes.value_counts())
```

### 1.2 Feature Engineering Toolkit

```python
# ── ENCODE CATEGORICALS ───────────────────────────────────────────────
cat_cols = train.select_dtypes('object').columns.tolist()
cat_cols = [c for c in cat_cols if c not in [TARGET, ID_COL]]

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

# ── FILL NULLS ─────────────────────────────────────────────────────────
num_cols = train.select_dtypes('number').columns.tolist()
num_cols = [c for c in num_cols if c not in [TARGET, ID_COL]]

train[num_cols] = train[num_cols].fillna(train[num_cols].median())
test[num_cols]  = test[num_cols].fillna(train[num_cols].median())

# ── INTERACTION FEATURES (add selectively) ────────────────────────────
# train['feat_ratio'] = train['col_a'] / (train['col_b'] + 1e-6)
# train['feat_product'] = train['col_a'] * train['col_b']

# ── AGGREGATION FEATURES (group by a key) ─────────────────────────────
# for col in cat_cols:
#     agg = train.groupby(col)['some_num'].agg(['mean','std','min','max'])
#     agg.columns = [f'{col}_{s}' for s in agg.columns]
#     train = train.join(agg, on=col)
#     test  = test.join(agg, on=col)

FEATURES = [c for c in train.columns if c not in [TARGET, ID_COL]]
X = train[FEATURES]
y = train[TARGET]
X_test = test[FEATURES]
```

### 1.3 LightGBM with K-Fold CV (your main model)

```python
# CLASSIFICATION
TASK = 'classification'  # or 'regression'
N_FOLDS = 5
SEED = 42

lgb_params = {
    'objective':       'binary',        # or 'multiclass', 'regression'
    'metric':          'auc',           # or 'multi_logloss', 'rmse'
    'n_estimators':    2000,
    'learning_rate':   0.05,
    'num_leaves':      63,
    'max_depth':       -1,
    'subsample':       0.8,
    'colsample_bytree':0.8,
    'min_child_samples': 20,
    'reg_alpha':       0.1,
    'reg_lambda':      0.1,
    'random_state':    SEED,
    'verbose':         -1,
    'n_jobs':          -1,
    # 'num_class': N,  # only for multiclass
    # 'device': 'gpu', # Kaggle GPU
}

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
# Use KFold for regression

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)  # or LGBMRegressor
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]  # or predict for regression
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

    score = roc_auc_score(y_val, oof_preds[val_idx])
    print(f'Fold {fold+1} AUC: {score:.4f}')

print(f'OOF AUC: {roc_auc_score(y, oof_preds):.4f}')

# ── SUBMIT ─────────────────────────────────────────────────────────────
sub[TARGET] = test_preds
# For multiclass: sub[TARGET] = np.argmax(test_preds_matrix, axis=1)
sub.to_csv('submission.csv', index=False)
```

### 1.4 XGBoost Drop-in (use for ensembling)

```python
import xgboost as xgb

xgb_params = {
    'objective':       'binary:logistic',
    'eval_metric':     'auc',
    'n_estimators':    2000,
    'learning_rate':   0.05,
    'max_depth':       6,
    'subsample':       0.8,
    'colsample_bytree':0.8,
    'use_label_encoder': False,
    'random_state':    42,
    'tree_method':     'hist',  # 'gpu_hist' on Kaggle GPU
}

model_xgb = xgb.XGBClassifier(**xgb_params)
model_xgb.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=100,
              verbose=200)
```

### 1.5 Multiclass Classification Template

```python
# Change lgb_params:
lgb_params['objective'] = 'multiclass'
lgb_params['metric'] = 'multi_logloss'
lgb_params['num_class'] = y.nunique()

# Predict: shape (n_samples, n_classes)
oof_preds  = np.zeros((len(train), y.nunique()))
test_preds = np.zeros((len(test), y.nunique()))

# In loop:
oof_preds[val_idx]  = model.predict_proba(X_val)
test_preds          += model.predict_proba(X_test) / N_FOLDS

# Final label:
sub[TARGET] = np.argmax(test_preds, axis=1)

# If label was string-encoded earlier:
# sub[TARGET] = le.inverse_transform(np.argmax(test_preds, axis=1))
```

### 1.6 Regression Template Changes

```python
lgb_params['objective'] = 'regression'
lgb_params['metric']    = 'rmse'

oof_preds = np.zeros(len(train))
# In loop: model.predict(X_val), model.predict(X_test)
# Score: np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
```

---

## 📝 PART 2 — NLP

> Rule: **No pretrained model.** Build your own embedding → RNN → head.
> Exception: **TF-IDF + classical ML is perfectly valid** for classification (often beats LSTM on small data in 5 hours!).

### 2.1 Fast Baseline: TF-IDF + LogReg (get on the board in 10 min)

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp

train = pd.read_csv('/kaggle/input/.../train.csv')
test  = pd.read_csv('/kaggle/input/.../test.csv')
sub   = pd.read_csv('/kaggle/input/.../sample_submission.csv')

TEXT_COL = 'text'
TARGET   = 'label'

# ── CLEAN ─────────────────────────────────────────────────────────────
import re
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train[TEXT_COL] = train[TEXT_COL].apply(clean)
test[TEXT_COL]  = test[TEXT_COL].apply(clean)

# ── VECTORIZE ─────────────────────────────────────────────────────────
tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True,
    min_df=2,
)
tfidf_char = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 4),
    max_features=30000,
    sublinear_tf=True,
    min_df=2,
)

X_word_tr = tfidf_word.fit_transform(train[TEXT_COL])
X_char_tr = tfidf_char.fit_transform(train[TEXT_COL])
X_word_te = tfidf_word.transform(test[TEXT_COL])
X_char_te = tfidf_char.transform(test[TEXT_COL])

X_tr = sp.hstack([X_word_tr, X_char_tr])
X_te = sp.hstack([X_word_te, X_char_te])

le = LabelEncoder()
y = le.fit_transform(train[TARGET])

# ── TRAIN ─────────────────────────────────────────────────────────────
model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', multi_class='auto')
scores = cross_val_score(model, X_tr, y, cv=5, scoring='f1_macro')
print(f'CV F1: {scores.mean():.4f} ± {scores.std():.4f}')

model.fit(X_tr, y)
preds = le.inverse_transform(model.predict(X_te))
sub[TARGET] = preds
sub.to_csv('submission_tfidf.csv', index=False)
```

### 2.2 LSTM Text Classifier (from scratch, PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

# ── TOKENIZER + VOCAB ─────────────────────────────────────────────────
def tokenize(text): return str(text).lower().split()

def build_vocab(texts, min_freq=2, max_vocab=30000):
    counter = Counter(tok for t in texts for tok in tokenize(t))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.most_common(max_vocab):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(train[TEXT_COL].tolist())
VOCAB_SIZE = len(vocab)
PAD_IDX = 0

def encode(text, vocab, max_len=128):
    toks = tokenize(text)[:max_len]
    ids  = [vocab.get(t, 1) for t in toks]
    pad  = [PAD_IDX] * (max_len - len(ids))
    return ids + pad

# ── DATASET ────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, vocab=None, max_len=128):
        self.X = [encode(t, vocab, max_len) for t in texts]
        self.y = labels
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.long)
        if self.y is not None:
            return x, torch.tensor(self.y[i], dtype=torch.long)
        return x

N_CLASSES = y.nunique() if hasattr(y, 'nunique') else len(set(y))

train_ds = TextDataset(train[TEXT_COL].tolist(), list(y_encoded), vocab)
test_ds  = TextDataset(test[TEXT_COL].tolist(), vocab=vocab)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# ── MODEL ──────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes,
                 n_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if n_layers > 1 else 0)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        emb = self.drop(self.embed(x))           # (B, L, E)
        out, (hn, _) = self.lstm(emb)            # out: (B, L, 2H)
        # Use last hidden from both directions
        h = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, 2H)
        return self.fc(self.drop(h))             # (B, C)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(VOCAB_SIZE, embed_dim=128, hidden_dim=256,
                       n_classes=N_CLASSES).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ── TRAIN LOOP ─────────────────────────────────────────────────────────
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
    print(f'Epoch {epoch+1} done')

# ── INFERENCE ──────────────────────────────────────────────────────────
model.eval()
all_preds = []
with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(device)
        preds = model(xb).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

sub[TARGET] = le.inverse_transform(all_preds)
sub.to_csv('submission_lstm.csv', index=False)
```

### 2.3 Token-Level Classification (NER / POS tagging)

> Use BIO tagging scheme: B-ENT, I-ENT, O

```python
# ── DATA FORMAT ASSUMPTION ─────────────────────────────────────────────
# Each row: token, tag  (CoNLL style)
# Or: sentence_id, token, tag

# Build word & tag vocabs
def build_tag_vocab(tags_list):
    tags = {t for seq in tags_list for t in seq}
    return {t: i for i, t in enumerate(sorted(tags))}

# ── MODEL: BiLSTM Tagger ───────────────────────────────────────────────
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_tags,
                 n_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if n_layers > 1 else 0)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden_dim * 2, n_tags)

    def forward(self, x):
        # x: (B, L)
        emb = self.drop(self.embed(x))         # (B, L, E)
        out, _ = self.lstm(emb)                # (B, L, 2H)
        return self.fc(self.drop(out))         # (B, L, T)

# Loss: ignore padding
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG_IDX)

# Training same as above but targets are (B, L) shaped
# Prediction: argmax over last dim → decode with tag_vocab inverse

# ── EVALUATION (seqeval) ───────────────────────────────────────────────
# pip install seqeval (available on Kaggle)
from seqeval.metrics import f1_score, classification_report
# f1_score(true_sequences, pred_sequences, average='micro')
```

---

## 🖼️ PART 3 — COMPUTER VISION

> Rule: **No pretrained model.** Build a 2D CNN from scratch.
> Key trick: **Sub-image/patch cropping** is explicitly in the kisi-kisi. Do it first.

### 3.1 Image Loading + Patch Pipeline

```python
import cv2
import numpy as np
import os
from glob import glob

# ── LOAD & RESIZE ─────────────────────────────────────────────────────
def load_image(path, size=(128, 128)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

# ── PATCH CROPPING (sub-image segmentation) ────────────────────────────
def extract_patches(img, patch_size=64, stride=32):
    H, W, C = img.shape
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches  # list of (patch_size, patch_size, 3)

# ── AUGMENTATION ──────────────────────────────────────────────────────
def augment(img):
    # Horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    # Brightness jitter
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(float)
    hsv[..., 2] *= np.random.uniform(0.7, 1.3)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # Gaussian blur (mild)
    if np.random.rand() > 0.7:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# ── NORMALIZE ─────────────────────────────────────────────────────────
def normalize(img):
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std
```

### 3.2 PyTorch Dataset + DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, paths, labels=None, size=(128, 128), augment_fn=None):
        self.paths     = paths
        self.labels    = labels
        self.size      = size
        self.augment   = augment_fn

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = load_image(self.paths[i], self.size)
        if self.augment and self.labels is not None:
            img = self.augment(img)
        img = normalize(img)
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        if self.labels is not None:
            return img, torch.tensor(self.labels[i], dtype=torch.long)
        return img
```

### 3.3 2D CNN from Scratch

```python
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel//2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class SimpleCNN(nn.Module):
    def __init__(self, n_classes, in_channels=3, img_size=128):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),      # 128→64
            ConvBlock(32, 64),               # 64→32
            ConvBlock(64, 128),              # 32→16
            ConvBlock(128, 256),             # 16→8
            ConvBlock(256, 256, pool=False), # 8→8 (no pool)
            nn.AdaptiveAvgPool2d((4, 4)),    # → 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(n_classes=N_CLASSES).to(device)
```

### 3.4 Training Loop

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
EPOCHS = 20

for epoch in range(EPOCHS):
    # ── TRAIN
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
    scheduler.step()

    # ── VALIDATE
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds  = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += len(yb)
    val_acc = correct / total
    print(f'Epoch {epoch+1:02d} | Val Acc: {val_acc:.4f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

# Reload best
model.load_state_dict(torch.load('best_model.pth'))

# ── INFERENCE ──────────────────────────────────────────────────────────
model.eval()
all_preds = []
with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(device)
        preds = model(xb).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

sub[TARGET] = all_preds
sub.to_csv('submission_cnn.csv', index=False)
```

### 3.5 Test-Time Augmentation (TTA) — quick score boost

```python
def predict_tta(model, loader, n_tta=5):
    model.eval()
    all_probs = []
    for _ in range(n_tta):
        probs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                p = torch.softmax(model(xb), dim=1).cpu().numpy()
                probs.append(p)
        all_probs.append(np.vstack(probs))
    return np.mean(all_probs, axis=0)  # average over TTA runs
```

---

## 🔀 PART 4 — ENSEMBLING & SUBMISSION

### 4.1 Simple Averaging (always try this)

```python
# For classification (probabilities)
ensemble = (pred_lgb + pred_xgb) / 2
# or weighted:
ensemble = 0.6 * pred_lgb + 0.4 * pred_xgb

# For regression
ensemble = (pred_lgb + pred_xgb) / 2
```

### 4.2 Rank Averaging (more robust)

```python
from scipy.stats import rankdata
ranked = (rankdata(pred_lgb) + rankdata(pred_xgb)) / 2
# normalise back to [0,1]
final = (ranked - ranked.min()) / (ranked.max() - ranked.min())
```

### 4.3 Submission Checklist

```python
# 1. Check format
print(sub.head())
print(sub.shape)
print(sub[TARGET].value_counts())

# 2. No NaN
assert sub[TARGET].isnull().sum() == 0

# 3. Match test size
assert len(sub) == len(test)

# 4. Save
sub.to_csv('submission_final.csv', index=False)
```

---

## ⏱️ TIME MANAGEMENT (5-hour plan for 3 people)

| Time | Person A (Tabular) | Person B (NLP) | Person C (CV) |
|---|---|---|---|
| 0:00–0:20 | Download data, EDA | Download data, EDA | Download data, EDA |
| 0:20–1:00 | Baseline LightGBM | TF-IDF + LogReg | CNN skeleton ready |
| 1:00–2:00 | Feature engineering | LSTM training | Augmentation + training |
| 2:00–3:00 | Hyperparameter tune | LSTM tune or ensemble TF-IDF+LSTM | TTA, LR schedule |
| 3:00–4:00 | XGBoost baseline + ensemble | Token classification (if needed) | Deeper CNN or patch-level voting |
| 4:00–4:30 | Final submission | Final submission | Final submission |
| 4:30–5:00 | Source code packaging for email | → | → |

---

## 🚨 QUICK REFERENCE — Common Pitfalls

| Problem | Fix |
|---|---|
| LB score << CV score | Use StratifiedKFold, don't shuffle before split |
| LGBM overfitting | Increase `min_child_samples`, lower `num_leaves` |
| LSTM not converging | Lower LR (1e-4), clip gradients, check padding |
| CV image loading OOM | Reduce batch size, use `num_workers=2` in DataLoader |
| Multiclass label mismatch | Always use `LabelEncoder` and `inverse_transform` for submission |
| Slow CNN on CPU | Force Kaggle GPU: Settings → Accelerator → GPU T4 |
| NaN loss in CNN | Normalize images (divide by 255), check for all-black images |

---

## 📋 USEFUL KAGGLE NOTEBOOKS TO BOOKMARK (open before competition)

- **Tabular**: https://www.kaggle.com/code/dlarionov/feature-engineering-xgboost
- **NLP LSTM**: https://www.kaggle.com/code/arunrk7/nlp-beginner-text-classification-using-lstm
- **NLP NER BiLSTM-CRF**: https://www.kaggle.com/code/ab971631/ner-with-bi-lstm-crf
- **CV CNN PyTorch**: https://www.kaggle.com/code/shtrausslearning/pytorch-cnn-binary-image-classification
- **General LGBM template**: https://www.kaggle.com/code/sudalairajkumar/winning-solutions-of-kaggle-competitions

---

## 🔑 KEY IMPORTS CHEATSHEET (copy to top of every notebook)

```python
# ── STANDARD ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

# ── SKLEARN ────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             mean_squared_error, classification_report)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

# ── GBDT ──────────────────────────────────────────────────────────────
import lightgbm as lgb
import xgboost as xgb

# ── DEEP LEARNING ─────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── CV ────────────────────────────────────────────────────────────────
import cv2

# ── NLP ───────────────────────────────────────────────────────────────
import re
from collections import Counter
```

---

*Good luck tomorrow. First submission by T+30 min. Iterate fast. Source code email by 14:00.*