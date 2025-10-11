# main_embeddings_auto_supervised.py
# Visualisation + clustering avec sortie en fichiers HTML (Plotly)

import os
import datetime as dt
import torch
import torchtext
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px
from RNN_Model_auto_supervised import CustomRNN_manual

# Autoriser Vocab si présent dans le checkpoint (utile avec PyTorch >= 2.6)
torch.serialization.add_safe_globals([torchtext.vocab.Vocab])

# ---------- Paramètres simples ----------
CKPT_PATH = "rnn_model_checkpoint_auto_supervised.pth"
OUT_DIR   = "viz_embeddings"
NUM_TOO_FREQ = 150   # sauter les tokens les plus fréquents
NUM_WORDS    = 400   # nombre de mots à afficher
PERPLEXITY   = 30
DBSCAN_EPS   = 4.0
DBSCAN_MIN   = 5
# ---------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

# ---------- Charger modèle ----------
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
vocab = ckpt["vocab"]
classes = ckpt.get("classes", vocab)
state = ckpt["model_state_dict"]
pad_idx  = ckpt.get("pad_idx",  getattr(vocab, "stoi", {}).get("<pad>", 0))
mask_idx = ckpt.get("mask_idx", getattr(vocab, "stoi", {}).get("<mask>", None))
if mask_idx is None:
    raise ValueError("mask_idx introuvable (assure-toi d'avoir sauvegardé '<mask>').")

model = CustomRNN_manual(
    input_size=len(vocab), emb_size=128, hidden_size=128,
    output_size=len(classes), pad_idx=pad_idx, mask_idx=mask_idx
)
model.load_state_dict(state, strict=True)
model.eval()

# ---------- Embeddings ----------
emb = model.embedding.weight.detach().cpu().numpy()
print("Shape des embeddings :", emb.shape)

# ---------- t-SNE ----------
print("→ t-SNE...")
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, max_iter=1000, init="random", learning_rate="auto")
emb2d = tsne.fit_transform(emb)
print("→ OK.")

# ---------- Sous-ensemble pour affichage ----------
start = NUM_TOO_FREQ
end   = min(start + NUM_WORDS, len(vocab))
words = list(vocab.itos)[start:end]
X2d   = emb2d[start:end]
x, y  = X2d[:, 0], X2d[:, 1]

# ---------- Clustering ----------
def estimate_k(X, k_min=2, k_max=10):
    inertias = []
    Ks = range(k_min, k_max + 1)
    for k in Ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
    diffs = np.diff(inertias)
    # repère elbow grossier : plus petite variation absolue
    k_opt = Ks[np.argmin(np.abs(diffs))] if len(diffs) > 0 else k_min
    return k_opt

print("→ Estimation k (KMeans)...")
k_opt = estimate_k(X2d)
print(f"k ≈ {k_opt}")

kmeans = KMeans(n_clusters=k_opt, n_init=20, random_state=42)
labels_km = kmeans.fit_predict(X2d)

dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN)
labels_db = dbscan.fit_predict(X2d)

# ---------- Figures Plotly (écriture HTML) ----------
def save_scatter(filename, x, y, labels, words, title):
    fig = px.scatter(
        x=x, y=y, text=words, color=labels.astype(str),
        title=title, width=950, height=700
    )
    fig.update_traces(
        marker=dict(size=8, opacity=0.8),
        textposition="top center",
        hovertemplate="Mot : %{text}<br>Cluster : %{marker.color}<extra></extra>"
    )
    path = os.path.join(OUT_DIR, filename)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    print(f"→ Fichier écrit : {path}")

save_scatter(
    f"tsne_kmeans_k{k_opt}_{stamp}.html",
    x, y, labels_km, words,
    title=f"t-SNE + KMeans (k={k_opt}) sur embeddings de mots"
)

save_scatter(
    f"tsne_dbscan_eps{DBSCAN_EPS}_min{DBSCAN_MIN}_{stamp}.html",
    x, y, labels_db, words,
    title=f"t-SNE + DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN}) sur embeddings de mots"
)
