import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('stopwords')

# Read the data
df = pd.read_csv('news_data\MSFT_news_20250109.csv')

# Combine all titles into one text
text = ' '.join(df['title'].astype(str))

# Clean the text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

text = clean_text(text)

# Get stop words and add custom ones
stop_words = set(stopwords.words('english'))
custom_stops = {'microsoft', 'msft', 'corp', 'corporation'}
stop_words.update(custom_stops)

# Create and generate word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=stop_words,
    min_font_size=10,
    max_font_size=150
).generate(text)

# Create figure and display word cloud
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Microsoft News Headlines')
plt.show()

# Get word frequencies (excluding stop words)
words = text.split()
words = [word for word in words if word not in stop_words]
word_freq = Counter(words).most_common(10)

print("\nTop 10 most frequent words:")
for word, freq in word_freq:
    print(f"{word}: {freq}")

# Basic statistics about the headlines
print("\nHeadline Statistics:")
print(f"Total number of headlines: {len(df)}")
print(f"Average headline length (characters): {df['title'].str.len().mean():.1f}")
print(f"Average word count per headline: {df['title'].str.split().str.len().mean():.1f}")