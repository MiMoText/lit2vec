from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Path to your 'all.tsv' file
file_path = 'all.tsv'

# Prepare sentences for training the model
sentences = LineSentence(file_path)

# Train a Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Save the trained model
model.save("word2vec.model")