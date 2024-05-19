import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
# Path to the dataset
file_path= "../training_set_rel3.tsv"


# Load the dataset with an alternative encoding
essays = pd.read_csv(file_path, delimiter='\t', encoding='latin1')

# Check the first few rows of the dataset
print(essays.head())

relevant_columns = ['essay_id', 'essay', 'domain1_score']  # Add or remove column names as necessary

# Filter the dataset to include only the relevant columns
essays = essays[relevant_columns]

# Check the first few rows of the dataset to confirm the correct columns are included
essays.head()

save_path = '../essay.csv'
# Save the filtered dataset to a new CSV file
essays.to_csv(save_path, index=False)
# Load the new dataset
filtered_essays = pd.read_csv(save_path)

# Print the first few rows to confirm the content
print(filtered_essays.head())

filtered_essays.info()

essays=filtered_essays

# Overview of the dataset's structure and missing values
print("Dataset Info:")
print(essays.info())
print("\nMissing Value Counts:")
print(essays.isnull().sum())

# Check for duplicates
duplicate_count = essays.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicate_count}")

# Basic statistics for numerical columns
print("\nBasic Statistics for Numerical Columns:")
print(essays.describe())

# Check for unique counts in the 'essay_id' to confirm unique IDs
unique_essay_count = essays['essay_id'].nunique()
total_essay_count = len(essays)
print(f"\nTotal Unique Essay IDs: {unique_essay_count} out of {total_essay_count} total rows")


# Rename 'domain1_score' to 'score'
essays.rename(columns={'domain1_score': 'score'}, inplace=True)

# 1. Word Count Analysis
essays['word_count'] = essays['essay'].apply(lambda x: len(x.split()))
sns.histplot(essays['word_count'], bins=30)
plt.title("Distribution of Word Counts per Essay")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

# Correlation between word count and score
sns.scatterplot(x='word_count', y='score', data=essays)
plt.title("Word Count vs. Score")
plt.xlabel("Word Count")
plt.ylabel("Score")
plt.show()

# 2. Vocabulary Analysis
essays['unique_words'] = essays['essay'].apply(lambda x: len(set(x.split())))
sns.scatterplot(x='unique_words', y='score', data=essays)
plt.title("Unique Words vs. Score")
plt.xlabel("Unique Words")
plt.ylabel("Score")
plt.show()

# 3. Most Common Words
vectorizer = CountVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(essays['essay'])
common_words = vectorizer.get_feature_names_out()
word_counts = X.toarray().sum(axis=0)

sns.barplot(x=word_counts, y=common_words)
plt.title("Most Common Words in Essays")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

# 4. Sentiment Analysis
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

essays['sentiment'] = essays['essay'].apply(get_sentiment)
sns.histplot(essays['sentiment'], bins=30)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Polarity")
plt.ylabel("Frequency")
plt.show()

# Correlation between sentiment and score
sns.scatterplot(x='sentiment', y='score', data=essays)
plt.title("Sentiment vs. Score")
plt.xlabel("Sentiment Polarity")
plt.ylabel("Score")
plt.show()

"""1. Most Common Words in Essays
The bar chart indicates frequent usage of generic terms such as "people", "things", "think", "way", "time", which are common in discursive writing. Notably, the presence of "caps1" and "caps2" might be placeholders for anonymized names or locations, suggesting these are frequently mentioned in the essays. This information can guide us to possibly remove these placeholders during preprocessing to focus more on substantive content.

2. Sentiment Distribution
The histogram shows a normal-like distribution of sentiment polarity, centered slightly to the right, indicating a generally positive sentiment in the essays. This is typical for academic essays where affirmative statements are common. Understanding sentiment distribution can help in augmenting features that might correlate sentiment with essay scores.

3. Sentiment vs. Score
The scatter plot suggests a dense concentration of sentiment values around the neutral to slightly positive range, with varying scores. There doesn't appear to be a strong correlation between sentiment polarity and scores, indicating that while sentiment is an aspect of the text, it might not be a strong predictor of essay quality alone.

4. Unique Words vs. Score
This plot shows a positive trend, suggesting essays with a more diverse vocabulary tend to score higher. This supports the hypothesis that lexical diversity might be a good indicator of essay quality. This insight is particularly useful for models that analyze text complexity, such as LSTM and RNN, which can benefit from features that capture lexical richness.

5. Word Count vs. Score
There is a visible trend where longer essays tend to score higher, which is a common observation in educational settings as longer essays might demonstrate more detailed explanations and comprehensive coverage of a topic. This insight will be particularly useful for feature engineering, where essay length could be a predictive feature.

6. Distribution of Word Counts per Essay
The histogram shows that most essays have a word count in the range of about 150 to 350 words, with fewer essays being significantly shorter or longer. This provides a good understanding of the typical essay lengths and can help in standardizing input lengths for deep learning models or setting appropriate bins for categorization in machine learning models.
"""

