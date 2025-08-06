import pandas as pd

df = pd.read_csv('data_ratings_v2_improved.csv')

df.to_excel('data_ratings_v2_improved.xlsx', index=False)