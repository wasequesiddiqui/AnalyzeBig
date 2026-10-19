#%%
import config

#%%
print("Initial Facts Template defined:", bool(config.INITIAL_FACTS_TEMPLATE))
print("User Question Template defined:", bool(config.USER_QUESTION_TEMPLATE))
print("Chunk Size:", config.CHUNK_SIZE)
print("Similarity Top K:", config.SIMILARITY_TOP_K)

# %%
from profile_lookup import lookup_profile

profile_context = lookup_profile()
print(profile_context)

# %%
