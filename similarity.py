from gensim.models import Word2Vec

# Load the trained Word2Vec model
model = Word2Vec.load("word2vec.model")

# Initialize a list to store the words and results
word_list = []
results = []

# Read the words from the file
with open("work_identifier.tsv", "r") as f:
    for line in f:
        # Get the word from the line
        word = line.split('\t')[0]
        # Ensure the word is in the vocabulary
        if word in model.wv.key_to_index and word != "identifier":
            word_list.append(word)

# Compare each word with all other words
for i in range(len(word_list)):
    for j in range(i+1, len(word_list)):
        word1 = word_list[i]
        word2 = word_list[j]
        # Calculate the similarity between the two words
        similarity = model.wv.similarity(word1, word2)
        # Add the result to the list
        results.append((word1, word2, similarity))

# Sort the results by similarity in descending order
results.sort(key=lambda x: x[2], reverse=True)

# Write the results to a tsv file
with open("results.tsv", "w") as f:
    for word1, word2, similarity in results:
        f.write(f"{word1}\t{word2}\t{similarity}\n")
