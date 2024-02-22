import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn import metrics     # used for evaluating the performance of ML models


# transform text data into numerical representation for ML models
def build_feature_matrix(dataset, feature_type='frequency', ngram_range=(1, 1),
                         min_df=0.0, max_df=1.0):
    feature_type = feature_type.lower().strip()     # removing leading and trailing whitespaces

    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=min_df, max_df=max_df,
                                     ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df, max_df=max_df,
                                     ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                     ngram_range=ngram_range)
    else:
        raise Exception("Wrong feature type you entered. Possible values: "
                        "binary, frequency, tfidf")

    # Fit-transform the vectorizer to the text data and convert the result to float
    feature_matrix = vectorizer.fit_transform(dataset).astype(float)

    return vectorizer, feature_matrix


def display_evaluation_metrics(true_labels, predicted_labels, positive_class=1):

    print('Accuracy', np.round(metrics.accuracy_score(true_labels, predicted_labels), 2))

    print('Precision', np.round(metrics.precision_score(true_labels, predicted_labels,
                                                        pos_label=positive_class, average='binary'), 2))

    print('Recall', np.round(metrics.recall_score(true_labels, predicted_labels,
                                                  pos_label=positive_class, average='binary'), 2))

    print('F1 Score', np.round(metrics.f1_score(true_labels, predicted_labels,
                                                pos_label=positive_class, average='binary'), 2))

def display_confusion_matrix(true_labels, predicted_labels, classes=None):
    if classes is None:
        classes = [1, 0]
    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, labels=classes)

    # Create MultiIndex for both columns and index
    column_labels = pd.MultiIndex.from_product([['Predicted:'], classes])
    index_labels = pd.MultiIndex.from_product([['Actual:'], classes])

    # Create DataFrame with confusion matrix and MultiIndex
    cm_frame = pd.DataFrame(data=cm, columns=column_labels, index=index_labels)

    print(cm_frame)

