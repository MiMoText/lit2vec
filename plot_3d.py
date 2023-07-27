from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import plotly.graph_objects as go

# Load the trained Word2Vec model
model = Word2Vec.load("word2vec.model")

# Initialize a list to store the words
words = []

# Read the words from the file
with open("ne_list.txt", "r") as f:
    for line in f:
        word = line.strip()
        if word in model.wv.key_to_index:
            words.append(word)

# Get the word vectors
word_vectors = model.wv[words]

# Use t-SNE to reduce dimensionality
tsne = TSNE(n_components=3, random_state=0)
word_vectors_3d = tsne.fit_transform(word_vectors)

# Create a 3D scatter plot of the t-SNE output
fig = go.Figure(data=[go.Scatter3d(
    x=word_vectors_3d[:, 0],
    y=word_vectors_3d[:, 1],
    z=word_vectors_3d[:, 2],
    mode='markers+text',
    text=words,
    marker=dict(
        size=5,
        color=word_vectors_3d[:, 2],  # set color to z-axis value
        colorscale='Viridis',  # choose a colorscale
        opacity=0.8
    )
)])

# Set layout properties
fig.update_layout(title='Word2Vec t-SNE 3D Visualization',
                  scene=dict(xaxis=dict(title='Dimension 1'),
                             yaxis=dict(title='Dimension 2'),
                             zaxis=dict(title='Dimension 3')))

# Save the interactive plot as an HTML file
fig.write_html("interactive_plot_3d.html")
