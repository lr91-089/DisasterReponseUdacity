"""This module trains and fits an AdaBoost Model on the DisasterResponse.db"""

import sys
import pickle
import nltk

import pandas as pd

from sqlalchemy import create_engine

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

from text_utils import tokenize, StartingVerbExtractor


nltk.download(['punkt', 'wordnet', 'stopwords',
              'omw-1.4', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """Loads the dataframe from a database and creates input and target dataframes.

    Keyword arguments:
    database_filepath -- the filepath of the database to read the data from
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(
        "DisasterResponseTable",
        con=engine
    )
    X = df.message.values
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def build_model():
    """Returns a grid search object for an AdaBoostClassifier with custome transformers."""
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    parameters = {"clf__estimator__learning_rate": [0.9, 1.0],
                  }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model on the test set and prints a classification report.

    Keyword arguments:
    model -- the model to evaluate
    X_test -- the input data of the test set
    Y_test -- the target values of the test set
    category_names -- the multi output labels of the target values
    """
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    print(
        classification_report(
            Y_test.values,
            y_pred.values,
            target_names=category_names))


def save_model(model, model_filepath):
    """Stores the trained model into a pickle file.

    Keyword arguments:
    model -- the model to save
    model_filepath -- the filepath to save the model
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """This runs all functions at start within a procedure."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
