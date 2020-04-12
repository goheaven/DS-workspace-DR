import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk

#nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
#from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
# from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

#import pickle
from sklearn.externals import joblib


def load_data(database_filepath):
    """

    :param database_filepath: sqldatabase file path
    :return: X, Y and catagory names as labels
    """
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table('ETLResults', engine)
    df = df[df.related != 2]
    X = df.message
    y = df.iloc[:, 4:40]
    print(X.shape, y.shape, df.columns[4:])
    return X, y, df.columns[4:40]


def tokenize(text):
    """

    :param text: message from dataset
    :return: cleaned token
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """

    :return: GridSearchCV results
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        #        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=1)))
    ])

    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """

    :param model: trained model
    :param X_test: test dataset
    :param Y_test: test results dataset
    :param category_names: label
    """
    y_pred = model.predict(X_test)
    i = 0
    for index in category_names:
        print(index, classification_report(Y_test[index], y_pred[:, i]))
        i = i + 1


def save_model(model, model_filepath):
    """

    :param model: trained model
    :param model_filepath: pkl filepath
    """
    joblib.dump(model, model_filepath, compress=5)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
