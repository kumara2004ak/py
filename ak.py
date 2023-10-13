          import pandas as pd
          # Load the dataset
          df = pd.read_csv('Tweets.csv') 
         df.drop_duplicates(inplace=True)
          # Text preprocessing function
          def preprocess_text(text):
          # Your text preprocessing code here
         return text
         # Apply text preprocessing to the 'text' column
         df['text'] = df['text'].apply(preprocess_text)
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Visualize sentiment distribution
import matplotlib.pyplot as plt

plt.hist(df['sentiment_score'], bins=20)
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Sentiment Distribution')
plt.show()   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Tweets.csv')

# Display the first few rows of the dataset
print(df.head())

# Summary statistics of numerical columns
print(df.describe())

# Count the number of missing values in each column
print(df.isnull().sum())

# Visualize the distribution of sentiment
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.countplot(x='airline_sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Visualize sentiment distribution by airline
plt.figure(figsize=(10, 6))
sns.countplot(x='airline', hue='airline_sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution by Airline')
plt.xlabel('Airline')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment', loc='upper right')
plt.show()
from wordcloud import WordCloud
plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud of Tweets')
plt.show()
 import pandas as pd
import matplotlib.pyplot as plt

# Load the "Tweets.csv" dataset
df = pd.read_csv('Tweets.csv')

# Count the number of tweets in each sentiment category
sentiment_counts = df['airline_sentiment'].value_counts()

# Plot a bar chart
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution in Tweets.csv')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if needed
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Load the "Tweets.csv" dataset
df = pd.read_csv('Tweets.csv')

# Define colors for each sentiment category
colors = {'negative': 'red', 'neutral': 'gray', 'positive': 'green'}

# Count the number of tweets in each sentiment category
sentiment_counts = df['airline_sentiment'].value_counts()

# Plot a bar chart with different colors
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=[colors[sentiment] for sentiment in sentiment_counts.index])
plt.title('Sentiment Distribution in Tweets.csv')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if needed

# Create a legend to indicate colors
legend_labels = [plt.Rectangle((0, 0), 1, 1, fc=colors[sentiment]) for sentiment in sentiment_counts.index]
plt.legend(legend_labels, sentiment_counts.index)

plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the "Tweets.csv" dataset
df = pd.read_csv('Tweets.csv')

# Create a grouped bar chart with each airline as a category
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
sns.countplot(x='airline', hue='airline_sentiment', data=df, palette='viridis')

plt.title('Sentiment Distribution by Airline in Tweets.csv')
plt.xlabel('Airline')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment', loc='upper right')

plt.show()
import pandas as pd
import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the "Tweets.csv" dataset
df = pd.read_csv('Tweets.csv')

# Combine all tweet texts into a single string
text = ' '.join(df['text'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud of Common Words in Tweets.csv')
plt.show()
print(df.columns)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the "Tweets.csv" dataset
df = pd.read_csv('Tweets.csv')

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap to visualize the correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()




