import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text_basic(text_val):
    text_val = '' if pd.isna(text_val) else str(text_val)
    text_val = text_val.lower()
    text_val = re.sub(r'http\S+|www\.\S+', ' url ', text_val)
    text_val = re.sub(r'\S+@\S+', ' email ', text_val)
    text_val = re.sub(r'[^a-z0-9\s]', ' ', text_val)
    text_val = re.sub(r'\s+', ' ', text_val).strip()
    return text_val

df = pd.read_csv('fake_job_postings.csv')

df = df.drop_duplicates().copy()

text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_cols:
    df[col] = df[col].apply(clean_text_basic)

cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
for col in cat_cols:
    df[col] = df[col].fillna('missing').astype(str).str.lower().str.strip()

df['location'] = df['location'].fillna('missing').astype(str).str.lower().str.strip()

df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1).str.replace(r'\s+', ' ', regex=True).str.strip()
df['text_length'] = df['combined_text'].str.len()
df['word_count'] = df['combined_text'].str.split().str.len()
df['fraudulent'] = df['fraudulent'].astype(int)

df.to_csv('fake_job_postings_cleaned.csv', index=False)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['fraudulent']
)

train_df.to_csv('train_fake_jobs.csv', index=False)
test_df.to_csv('test_fake_jobs.csv', index=False)

print('Cleaning complete')
print('Created fake_job_postings_cleaned.csv')
print('Created train_fake_jobs.csv')
print('Created test_fake_jobs.csv')