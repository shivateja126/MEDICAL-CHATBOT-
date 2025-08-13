import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the JSON file with mappings
with open('mapping.json', 'r') as file:
    mappings = json.load(file)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

def doc_retrieve(query):
    # Compute the query embedding
    encoded_input = tokenizer(query, return_tensors='pt')
    query_embedding = model(**encoded_input).last_hidden_state.mean(dim=1)
    
    # Initialize variables to track the highest similarity and corresponding location
    max_similarity = -1
    most_similar_location = None

    # Loop through each entry in mappings to calculate similarity
    for entry in mappings:
        # Ensure the vector is loaded as a tensor
        vector = torch.tensor(entry['vector']).unsqueeze(0)  # Add batch dimension

        # Calculate cosine similarity between query embedding and this vector
        similarity = cosine_similarity(query_embedding.detach().numpy(), vector.detach().numpy())[0][0]

        # Update max_similarity and location if this similarity is higher
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_location = entry['location']

    return most_similar_location
