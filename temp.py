# %%

import pandas as pd

df = pd.read_pickle("/root/recommendation-engine/results_df.pkl")

# %%

df["search_space_video_id"].nunique()

# %%
df["query_video_id"].nunique()
