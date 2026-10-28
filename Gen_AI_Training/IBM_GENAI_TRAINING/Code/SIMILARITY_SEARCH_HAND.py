#%%
import math
import numpy as np
import scipy
import torch
from sentence_transformers import SentenceTransformer

# %%
# Example documents
documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]

# %%
# Load a pre-trained model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# %%
# Generate embeddings
embeddings = model.encode(documents)

# %%
embeddings.shape

# %%
embeddings

#%%
def euclidean_distance_fn(vector1, vector2):
    squared_sum = sum((x - y) ** 2 for x, y in zip(vector1, vector2))
    return math.sqrt(squared_sum)

# %%
euclidean_distance_fn(embeddings[0], embeddings[1])

# %%
euclidean_distance_fn(embeddings[1], embeddings[0])

#%%
l2_dist_manual = np.zeros([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        l2_dist_manual[i,j] = euclidean_distance_fn(embeddings[i], embeddings[j])

l2_dist_manual
# %%
l2_dist_manual[0,1]
# %%
l2_dist_manual[1,0]
# %%
l2_dist_manual_improved = np.zeros([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        ### YOUR CODE GOES HERE ###
        l2_dist_manual_improved[i,j] = euclidean_distance_fn(embeddings[i], embeddings[j])
l2_dist_manual_improved
# %%
l2_dist_scipy = scipy.spatial.distance.cdist(embeddings, embeddings, 'euclidean')
l2_dist_scipy
# %%
np.allclose(l2_dist_manual, l2_dist_scipy)
# %%
