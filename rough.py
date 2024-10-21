l = "jdsh dfjh lhdf lkhdfg agffd "
l.strip()

"""
import pandas as pd
df = pd.DataFrame([
    {'A' : 1, 'B' : 2},
    {'A' : 1, 'B' : 2},
    {'A' : 1, 'B' : 2},
    {'A' : 1, 'B' : 2},
])
print(df)
def create_prompt(row : dict):
    return 85 , 2
df["chosen"] , df["rejected"] = df.apply(create_prompt, axis = 1, result_type='expand')
print(df)
"""