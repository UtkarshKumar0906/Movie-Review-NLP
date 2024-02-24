import nltk
nltk.download('punkt')
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('wordnet')

# Load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
import random
random.shuffle(documents)

# Preprocess the documents
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    return [lemmatizer.lemmatize(word.lower()) for word in text if word.isalpha() and word.lower() not in stop_words]

# Extract features using Bag-of-Words
all_words = [preprocess_text(words) for words, category in documents]
word_list = [word for sublist in all_words for word in sublist]
word_features = nltk.FreqDist(word_list).most_common(3000)

def document_features(document):
    document_words = set(document)
    features = {}
    for word, freq in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Extract features from the dataset
featuresets = [(document_features(movie_reviews), category) for (movie_reviews, category) in documents]

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluate the classifier
accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)

review = "This movie is fantastic! The acting was superb and the plot was engaging."
tokens = word_tokenize(review)
preprocessed_review = preprocess_text(tokens)
features = document_features(preprocessed_review)
sentiment = classifier.classify(features)
print("Sentiment:", sentiment)