import preprocessing as process
import numpy as np
from sklearn.linear_model import SGDClassifier
from utils import build_feature_matrix, display_evaluation_metrics, display_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# IMPORTANT: '0' Stands for NO and '1' stands for Yes

"""
ML:
-data preparation
-feature extraction
-features
-training data
-test/validation data
-model
"""


# preparing dataset for training and testing
# shuffle data to avoid bias
shuffled_data = shuffle(process.token_list, random_state=42)
train_data, test_data = train_test_split(shuffled_data, test_size=0.25, random_state=42)

train_messages = np.array(train_data['message to examine'])
train_sentiment = np.array(train_data['label (depression result)'])
test_messages = np.array(test_data['message to examine'])
test_sentiment = np.array(test_data['label (depression result)'])

# prepare sample dataset for experiments
sample_docs = [2112, 2330, 1322, 89, 50, 599, 2387, 1229]
sample_data = [(test_messages[index], test_sentiment[index]) for index in sample_docs]


# convert training messages to strings
norm_train_tweets = [str(tweet) for tweet in train_messages]
# Extract features using TF-IDF vectorization
vectorizer, train_feature = build_feature_matrix(dataset=norm_train_tweets, feature_type='tfidf',
                                                 ngram_range=(1, 1), min_df=0.0, max_df=1.0)

# Initialize SVM classifier
svm = SGDClassifier(loss='hinge', max_iter=200)
# Train the SVM classifier using the extracted features and corresponding sentiments
svm.fit(train_feature, train_sentiment)


# convert training messages to strings
norm_test_tweets = [str(tweet) for tweet in test_messages]
# Transform test messages into features using the same vectorizer
test_features = vectorizer.transform(norm_test_tweets)

"""
for doc_index in sample_docs:
    print('Review--')
    print(test_messages[doc_index])
    print('Actual Labeled Sentiment: ', test_sentiment[doc_index])
    doc_features = test_features[doc_index]
    predicted_sentiment = svm.predict(doc_features)[0]
    print('Predicted Sentiment: ', predicted_sentiment)
    print('\n')
"""

# predict the sentiment for test dataset messages tweets
predicted_sentiment = svm.predict(test_features)

# evaluate model predict performance
display_evaluation_metrics(true_labels=test_sentiment, predicted_labels=predicted_sentiment, positive_class=0)

display_confusion_matrix(true_labels=test_sentiment, predicted_labels=predicted_sentiment, classes=[0, 1])

