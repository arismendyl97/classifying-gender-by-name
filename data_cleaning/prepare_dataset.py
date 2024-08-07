# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# %%
# Function to replace accented vowels
def remove_accents(text):
    accents = 'áéíóúÁÉÍÓÚ'
    replacements = 'aeiouAEIOU'
    translation_table = str.maketrans(accents, replacements)
    return text.translate(translation_table)

# %%
seed = 42

# %%
source_1_1 = pd.read_csv('../dataset/source_1_1.csv')
source_1_1['excel'] = 'source_1_1'
source_1_2 = pd.read_csv('../dataset/source_1_2.csv')
source_1_2['excel'] = 'source_1_2'
source_1_3 = pd.read_csv('../dataset/source_1_3.csv')
source_1_3['excel'] = 'source_1_3'
source_1_4 = pd.read_csv('../dataset/source_1_4.csv')
source_1_4['excel'] = 'source_1_4'
source_1_5 = pd.read_csv('../dataset/source_1_5.csv')
source_1_5['excel'] = 'source_1_5'

source_2_1 = pd.read_csv('../dataset/source_2_1.csv',delimiter=';')
people_df = source_2_1[['nombre','sexo_id']].rename(columns={"nombre": "name", "sexo_id": "gender"})
people_df["gender"] = people_df["gender"].apply(lambda x: 'F' if x == 1.0 else 'M' if x == 2.0 else '?')
people_df = people_df[people_df["gender"]!='?']
people_df['code'] = 'ES'
people_df['excel'] = 'people_df'

# %%
df = pd.concat([source_1_1,source_1_2,source_1_3,source_1_4,source_1_5,people_df], ignore_index=True, sort=False)

# %%
# Define the regex pattern to identify non-letter characters
pattern = r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ ]'

# Filter the DataFrame to keep only rows without non-letter characters, handling NaN values
df = df[df['name'].notna()]

df = df[
    (df['code'].isin(['ES', 'CO', 'PE', 'CL'])) &
    (df['name'] != '') &    # Ensure 'name' is not an empty string
    (df['gender'].isin(['M', 'F'])) &    # Ensure 'name' is not an empty string
    (~df['name'].str.contains(pattern))  # Apply regex pattern
]

df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)
df['name'] = df['name'].apply(lambda x: remove_accents(x))

# Drop duplicates and reset index
df['name'] = df['name'].str.lower()
df = df[['name','gender']].drop_duplicates(subset=['name','gender']).reset_index(drop=True)
df = df.sort_values(by=['gender', 'name'])

# %%
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed)

# Split the temporary set into validation (10%) and temp test set (20%)
val_df, temp_test_df = train_test_split(temp_df, test_size=2/3, random_state=seed)

# Split the temporary testing set into post-test set (5%) and test set (15%)
test_df, post_test_df = train_test_split(temp_test_df, test_size=1/4, random_state=seed)

# Display the sizes of each set
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Testing set size: {len(test_df)}")
print(f"Post Testing set size: {len(post_test_df)}")

# %%
train_df.to_csv('spanish names db - training.csv', index=False)
val_df.to_csv('spanish names db - validation.csv', index=False)
test_df.to_csv('spanish names db - testing.csv', index=False)
post_test_df.to_csv('spanish names db - post_testing.csv', index=False)


