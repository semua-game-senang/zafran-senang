"""
NLP PIPELINE — Text Classification
Dataset : 20 Newsgroups (sklearn built-in, no download needed)
Mirrors  : Any Kaggle text classification task
Tracks   : TF-IDF + LogReg baseline  →  BiLSTM from scratch (PyTorch)

Run: python nlp_pipeline.py
     (swap STAGE = 'tfidf' | 'lstm' at the bottom)
"""

import re
import numpy as np
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — Load Data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 0 : Loading 20 Newsgroups dataset")
print("=" * 60)

from sklearn.datasets import fetch_20newsgroups

# We use 4 categories to keep training fast; swap for None to get all 20
CATEGORIES = ['sci.space', 'rec.sport.hockey', 'talk.politics.guns', 'comp.graphics']

raw_train = fetch_20newsgroups(subset='train', categories=CATEGORIES,
                               remove=('headers', 'footers', 'quotes'))
raw_test  = fetch_20newsgroups(subset='test',  categories=CATEGORIES,
                               remove=('headers', 'footers', 'quotes'))

texts_train, y_train = raw_train.data, np.array(raw_train.target)
texts_test,  y_test  = raw_test.data,  np.array(raw_test.target)
label_names           = raw_train.target_names

print(f"  Train samples : {len(texts_train)}")
print(f"  Test  samples : {len(texts_test)}")
print(f"  Classes       : {label_names}")
print(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"\n  Example text (truncated):\n  '{texts_train[0][:200]}...'\n")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 1 : Cleaning text")
print("=" * 60)

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # keep only alphanumeric + spaces
    text = re.sub(r'\s+', ' ', text).strip()     # collapse whitespace
    return text

texts_train_clean = [clean(t) for t in texts_train]
texts_test_clean  = [clean(t) for t in texts_test]

print(f"  Before: '{texts_train[0][:120]}'")
print(f"  After : '{texts_train_clean[0][:120]}'")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2A — TF-IDF Baseline (get on the board in 10 min)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 2A : TF-IDF + Logistic Regression baseline")
print("=" * 60)
print("""
  WHY TF-IDF?
  Term Frequency–Inverse Document Frequency down-weights common words
  and up-weights rare, discriminative words. It turns each document
  into a sparse vector of length = vocab_size.

  TF(t,d)  = count(t in d) / total_words(d)
  IDF(t)   = log( N / df(t) )   [df = #docs containing t]
  TFIDF    = TF * IDF

  We use TWO vectorizers and concatenate them:
    - word n-grams (1,2): captures unigrams and bigrams
    - char n-grams (2,4): catches typos / morphology / compound words
""")

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Word-level TF-IDF
tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2),     # unigrams + bigrams
    max_features=50_000,    # keep top 50k by document frequency
    sublinear_tf=True,      # use log(1+tf) instead of raw tf
    min_df=2,               # ignore terms appearing in < 2 docs
)

# Character-level TF-IDF
tfidf_char = TfidfVectorizer(
    analyzer='char_wb',     # char n-grams within word boundaries
    ngram_range=(2, 4),
    max_features=30_000,
    sublinear_tf=True,
    min_df=2,
)

print("  Fitting word TF-IDF...")
X_word_tr = tfidf_word.fit_transform(texts_train_clean)
X_word_te = tfidf_word.transform(texts_test_clean)

print("  Fitting char TF-IDF...")
X_char_tr = tfidf_char.fit_transform(texts_train_clean)
X_char_te = tfidf_char.transform(texts_test_clean)

# Horizontally stack → one big sparse matrix per split
X_tr = sp.hstack([X_word_tr, X_char_tr])
X_te = sp.hstack([X_word_te, X_char_te])

print(f"  Train feature matrix shape : {X_tr.shape}  (samples × features)")
print(f"  Sparsity                   : {1 - X_tr.nnz / np.prod(X_tr.shape):.4f}  (most entries = 0)")
print()

print("  Training Logistic Regression...")
logreg = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', multi_class='auto')
logreg.fit(X_tr, y_train)

preds_tfidf = logreg.predict(X_te)
acc_tfidf   = accuracy_score(y_test, preds_tfidf)

print(f"\n  Test Accuracy (TF-IDF + LR): {acc_tfidf:.4f}")
print()
print(classification_report(y_test, preds_tfidf, target_names=label_names))

# Top predictive words per class
print("  Top 10 words per class:")
feature_names = tfidf_word.get_feature_names_out()
for cls_idx, cls_name in enumerate(label_names):
    coef = logreg.coef_[cls_idx][:len(feature_names)]  # word portion of coef
    top  = np.argsort(coef)[-10:][::-1]
    print(f"    {cls_name}: {[feature_names[i] for i in top]}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2B — BiLSTM from Scratch (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 2B : BiLSTM Text Classifier (from scratch, PyTorch)")
print("=" * 60)
print("""
  ARCHITECTURE:
    Input tokens (integer indices)
      → Embedding layer  [vocab_size × embed_dim]
         Learns a dense vector for each word. Think of it as a lookup table.
         Initialized randomly, updated during backprop.
      → Dropout
      → Bi-directional LSTM  [2 layers, hidden=256, bidirectional]
         At each timestep, the forward LSTM reads left→right,
         backward LSTM reads right→left.  Output: hidden of size 2×hidden_dim.
         We take the LAST hidden state of both directions and concatenate.
      → Dropout
      → Linear(2×hidden_dim → n_classes)
      → (CrossEntropyLoss includes softmax internally)
""")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN  = 256   # truncate/pad each document to this many tokens
EMBED_DIM   = 128
HIDDEN_DIM  = 256
N_LAYERS    = 2
DROPOUT     = 0.3
BATCH_SIZE  = 64
EPOCHS      = 5
LR          = 1e-3

print(f"  Device: {DEVICE}")

# ── Vocabulary ────────────────────────────────────────────────────────────────
def tokenize(text: str):
    return text.split()   # simple whitespace tokenizer (post-clean)

def build_vocab(texts, min_freq=2, max_vocab=30_000):
    counter = Counter(tok for t in texts for tok in tokenize(t))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.most_common(max_vocab):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(texts_train_clean)
VOCAB_SIZE = len(vocab)
PAD_IDX    = 0

print(f"  Vocabulary size : {VOCAB_SIZE:,} tokens")
print(f"  (words appearing < 2 times are mapped to <UNK>)")

def encode(text: str, vocab: dict, max_len: int = MAX_LEN):
    toks = tokenize(text)[:max_len]
    ids  = [vocab.get(t, 1) for t in toks]        # 1 = <UNK>
    pad  = [PAD_IDX] * (max_len - len(ids))        # right-pad with 0
    return ids + pad

# Show encoding of one example
sample_enc = encode(texts_train_clean[0], vocab)
print(f"\n  Sample encoding (first 20 token IDs): {sample_enc[:20]}")
print(f"  Decoded back: {[list(vocab.keys())[list(vocab.values()).index(i)] if i in vocab.values() else '<PAD>' for i in sample_enc[:20]]}")
print()

# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.X = [encode(t, vocab, max_len) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (torch.tensor(self.X[i], dtype=torch.long),
                torch.tensor(self.y[i],  dtype=torch.long))

le = LabelEncoder()
y_tr_enc = le.fit_transform(y_train)
y_te_enc = le.transform(y_test)
N_CLASSES = len(le.classes_)

train_ds = TextDataset(texts_train_clean, y_tr_enc, vocab)
test_ds  = TextDataset(texts_test_clean,  y_te_enc, vocab)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train batches : {len(train_loader)}  (batch_size={BATCH_SIZE})")
print(f"  Test  batches : {len(test_loader)}")

# Show one batch shape
xb, yb = next(iter(train_loader))
print(f"  Batch X shape : {xb.shape}  → (batch, seq_len)")
print(f"  Batch y shape : {yb.shape}  → (batch,)")
print()

# ── Model ─────────────────────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes,
                 n_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()

        # Embedding: integer token → dense vector
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # LSTM: processes sequence left-to-right AND right-to-left simultaneously
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,       # input shape: (batch, seq, feature)
            bidirectional=True,     # doubles output hidden_dim
            dropout=dropout if n_layers > 1 else 0,
        )

        self.drop = nn.Dropout(dropout)

        # Final classifier: concatenated last-hidden from both directions
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        # x : (B, L)  — batch of token index sequences
        emb = self.drop(self.embed(x))          # (B, L, E)

        # out  : (B, L, 2H)  — hidden state at every timestep
        # hn   : (2*n_layers, B, H)  — final hidden states
        out, (hn, _) = self.lstm(emb)

        # hn[-2] = last layer, forward direction
        # hn[-1] = last layer, backward direction
        h = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, 2H)

        return self.fc(self.drop(h))             # (B, n_classes)


model = BiLSTMClassifier(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    n_classes=N_CLASSES,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

print("  Model architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Total trainable parameters: {total_params:,}")
print()

# ── Training ──────────────────────────────────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"  Training for {EPOCHS} epochs...")
print(f"  {'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Acc':>7}")
print("  " + "-" * 42)

for epoch in range(1, EPOCHS + 1):
    # ── Train
    model.train()
    total_loss = correct = total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)          # (B, C) raw scores (no softmax yet)
        loss   = criterion(logits, yb)
        loss.backward()             # compute gradients
        optimizer.step()            # update weights

        total_loss += loss.item() * len(yb)
        preds       = logits.argmax(dim=1)
        correct    += (preds == yb).sum().item()
        total      += len(yb)

    train_loss = total_loss / total
    train_acc  = correct / total

    # ── Validate (on test set here for simplicity)
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb   = xb.to(DEVICE), yb.to(DEVICE)
            preds     = model(xb).argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total   += len(yb)
    val_acc = val_correct / val_total

    print(f"  {epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_acc:>7.4f}")

# ── Inference ─────────────────────────────────────────────────────────────────
print("\n  Running inference on test set...")
model.eval()
all_preds = []
all_probs = []

with torch.no_grad():
    for xb, _ in test_loader:
        xb     = xb.to(DEVICE)
        logits = model(xb)
        probs  = torch.softmax(logits, dim=1)   # convert logits → probabilities
        preds  = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

acc_lstm = accuracy_score(y_te_enc, all_preds)
print(f"\n  Test Accuracy (BiLSTM): {acc_lstm:.4f}")
print()
print(classification_report(y_te_enc, all_preds, target_names=label_names))

# ── Compare ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  TF-IDF + LogReg : {acc_tfidf:.4f}")
print(f"  BiLSTM (scratch): {acc_lstm:.4f}")
print("""
  Key takeaway:
    - TF-IDF is faster to train and often competitive on short text.
    - BiLSTM captures word ORDER and context; wins on longer sequences.
    - For competition: submit TF-IDF first (10 min), then improve with LSTM.
""")
