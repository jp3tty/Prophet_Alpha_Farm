import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import os
from datetime import datetime

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Set the file path
file_path = r'C:\Users\jpetty\Documents\Projects\dsWithDen\news_data\MSFT_news_20250108.csv'

# Verify file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

# Read the data
df = pd.read_csv(file_path)

# Custom function to parse the time
def parse_time(time_str):
    if 'Today' in time_str:
        time_str = time_str.replace('Today ', '')
    try:
        # Convert 12-hour format to 24-hour format
        return datetime.strptime(time_str, '%I:%M%p').hour
    except:
        return None

# Extract hour from the date string
df['hour'] = df['date'].apply(parse_time)

# Remove any rows where we couldn't parse the hour
df = df.dropna(subset=['hour'])

# 1. Time Analysis
plt.figure(figsize=(12, 6))
df['hour'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of News Articles by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Articles')
plt.show()

# 2. Sentiment Analysis
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

df['sentiment'] = df['title'].apply(get_sentiment)

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='sentiment', bins=20)
plt.title('Distribution of Headline Sentiment')
plt.xlabel('Sentiment Score (Negative to Positive)')
plt.show()

# 3. Headline Length Analysis
df['title_length'] = df['title'].str.len()
df['word_count'] = df['title'].str.split().str.len()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(data=df, x='title_length', bins=20, ax=ax1)
ax1.set_title('Distribution of Headline Lengths')
ax1.set_xlabel('Number of Characters')

sns.histplot(data=df, x='word_count', bins=20, ax=ax2)
ax2.set_title('Distribution of Word Counts')
ax2.set_xlabel('Number of Words')
plt.tight_layout()
plt.show()

# 4. Common Bigrams Analysis
def get_bigrams(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return list(ngrams(tokens, 2))

all_bigrams = []
for title in df['title']:
    all_bigrams.extend(get_bigrams(str(title)))

bigram_freq = Counter(all_bigrams).most_common(10)

# Plot bigram frequencies
plt.figure(figsize=(12, 6))
x, y = zip(*[(f"{bg[0]} {bg[1]}", freq) for bg, freq in bigram_freq])
plt.bar(x, y)
plt.xticks(rotation=45, ha='right')
plt.title('Most Common Bigrams in Headlines')
plt.tight_layout()
plt.show()

# 5. Part of Speech Analysis
def get_pos_tags(text):
    tokens = word_tokenize(str(text))
    return nltk.pos_tag(tokens)

pos_tags = []
for title in df['title']:
    pos_tags.extend([tag[1] for tag in get_pos_tags(title)])

pos_freq = Counter(pos_tags)

plt.figure(figsize=(12, 6))
plt.bar(pos_freq.keys(), pos_freq.values())
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Parts of Speech in Headlines')
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print("==================")
print(f"Total number of headlines: {len(df)}")
print(f"Average headline length: {df['title_length'].mean():.1f} characters")
print(f"Average word count: {df['word_count'].mean():.1f} words")
print(f"Average sentiment score: {df['sentiment'].mean():.3f}")
print("\nMost common bigrams:")
for bigram, count in bigram_freq[:5]:
    print(f"{bigram}: {count}")

# Print time distribution
print("\nNews Distribution by Hour:")
hour_dist = df['hour'].value_counts().sort_index()
for hour, count in hour_dist.items():
    print(f"{hour:02d}:00 - {count} articles")