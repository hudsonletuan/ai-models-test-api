# You need to install "datasets" and "sentence-transformers" packages first to have access to utilities from HuggingFace

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded = tokenizer.encode("Do not meddle in the affairs of wizards")

print("Encoded text:", tokenizer.convert_ids_to_tokens(encoded))
print(encoded)

encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input.keys())
print(encoded_input['input_ids'])

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

words = ["quick", "fast", "red", "blue", "ferari"]
single_word_embeddings = model.encode(words)

for word, embed in zip(words, single_word_embeddings):
  print("word: ", word)
  print("embed: ", embed[0:10])
  print("")

from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(single_word_embeddings)
print(cos_sim)

from scipy.spatial.distance import cosine
for sentence in sentences:
    print("Sentence:", sentence)
print("")
print("Cosine similarity between the first two sentences:", cosine(embeddings[0], embeddings[1]))
print("Cosine similarity between the second and third sentences:", cosine(embeddings[1], embeddings[2]))
print("Cosine similarity between the first and third sentences:", cosine(embeddings[0], embeddings[2]))

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Input sentence
sentence = "Time after time, time flies like an arrow, but fruit flies like a banana."

# Words of interest
words_of_interest = ["time", "flies", "like"]

# Tokenize the input sentence
tokens = tokenizer.tokenize(sentence)

# Initialize a dictionary to store the embeddings for each word
word_embeddings = {word: [] for word in words_of_interest}

# Disable gradient calculation for inference
with torch.no_grad():
    # Tokenize the input text and convert to a tensor
    encoded_input = tokenizer(sentence, return_tensors='pt')

    # Get the model's output (hidden states)
    outputs = model(**encoded_input)
    last_hidden_states = outputs.last_hidden_state

    # Extract the embeddings for the words of interest
    for i, token in enumerate(tokens):
        if token.lower() in words_of_interest:
            word_embeddings[token.lower()].append(last_hidden_states[0, i])

# Compute cosine similarities between the word embeddings
for word, embeddings in word_embeddings.items():
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))
            print(f"Cosine similarity between '{word}' in position {i + 1} and position {j + 1}: {similarity[0][0]}")

import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Define the sentences
sentence1 = "Time flies like an arrow"
sentence2 = "Fruit flies like a banana"

# Tokenize and encode the sentences
encoded_sentence1 = tokenizer(sentence1, return_tensors='pt')
encoded_sentence2 = tokenizer(sentence2, return_tensors='pt')

# Identify the 'like' token index
word = "like"
word_index1 = tokenizer.encode(sentence1, add_special_tokens=True).index(tokenizer.encode(word)[1]) - 1
word_index2 = tokenizer.encode(sentence2, add_special_tokens=True).index(tokenizer.encode(word)[1]) - 1

# Get the embeddings for each encoded sentence
with torch.no_grad(): # Disable gradient tracking
    output_sentence1 = model(**encoded_sentence1)
    output_sentence2 = model(**encoded_sentence2)

# Extract the embeddings for the word 'like' from both sentences using their indices
embedding_sentence1 = output_sentence1.last_hidden_state[0, word_index1]
embedding_sentence2 = output_sentence2.last_hidden_state[0, word_index2]

# Calculate the cosine similarity
similarity = 1 - cosine(embedding_sentence1.detach().numpy(), embedding_sentence2.detach().numpy())

# Print the results
print("Cosine Similarity:", similarity)

from transformers import BertTokenizer, BertModel
import torch

# Initialize the tokenizer and model from the BERT family
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sentences to analyze
sentence1 = "Time flies like an arrow"
sentence2 = "Fruit flies like a banana"

# Tokenize and encode sentences for BERT input
tokens1 = tokenizer.encode_plus(sentence1, return_tensors='pt')
tokens2 = tokenizer.encode_plus(sentence2, return_tensors='pt')

# Get the embeddings from the BERT model
outputs1 = model(**tokens1)
outputs2 = model(**tokens2)

# Retrieve the embeddings for the word "like" for each sentence
# Assuming 'like' is not the first word and does not get split into subwords.
like_index1 = tokens1['input_ids'][0].tolist().index(tokenizer.encode('like', add_special_tokens=False)[0])
like_index2 = tokens2['input_ids'][0].tolist().index(tokenizer.encode('like', add_special_tokens=False)[0])

like_embedding1 = outputs1.last_hidden_state[0, like_index1]
like_embedding2 = outputs2.last_hidden_state[0, like_index2]

# Compare the embeddings, e.g., by using cosine similarity
cosine_similarity = torch.nn.CosineSimilarity(dim=0)
similarity = cosine_similarity(like_embedding1, like_embedding2).item()

print(f"Cosine similarity between 'like' in both contexts: {similarity}")