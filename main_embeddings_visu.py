import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from RNN_Model import CustomRNN_manual
import plotly.express as px

# Chargement du modèle entraîné.
checkpoint = torch.load("rnn_model_checkpoint.pth", map_location=torch.device('cpu'))

vocab = checkpoint['vocab']
model_state = checkpoint['model_state_dict']

model = CustomRNN_manual(input_size=len(vocab), emb_size=128, hidden_size=128, output_size=len(checkpoint['classes']))
model.load_state_dict(model_state)
model.eval()

embeddings = model.embedding.weight.detach().numpy()
print("Shape des embeddings :", embeddings.shape)

#Reduction simple sur 2 dimensions pour visualisation simple (sur les X mots les plus utilisés) :
"""
#PCA : 
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings)
"""
#t-SNE
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
embeddings_2d_tsne = tsne.fit_transform(embeddings)

num_too_frequent = 150
num_words = 200
words = list(vocab.itos)[num_too_frequent:num_too_frequent+num_words]

x = embeddings_2d_tsne[num_too_frequent:num_too_frequent + num_words, 0]
y = embeddings_2d_tsne[num_too_frequent:num_too_frequent + num_words, 1]

fig = px.scatter(
    x=x,
    y=y,
    text=words,             # noms des mots au survol
    title="Projection 2D des embeddings de mots (t-SNE)",
    width=900,
    height=700
)

fig.update_traces(
    marker=dict(size=8, opacity=0.7, color="blue"),
    textposition="top center",
    hovertemplate="Mot : %{text}<extra></extra>"
)

fig.show(renderer="browser")