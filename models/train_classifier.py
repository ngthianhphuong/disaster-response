import sys
from sklearn.externals import joblib

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, ne_chunk

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''Function to load data from database, split it into features X and target y.
    Args:
        database_filepath: path to database file.
    Returns:
        X: 2D array of features values
        y: 2D array of target values
        category_names: list of categories in the dataset
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, engine)
    X = df.message.values
    y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns.tolist()
    return X, y, category_names

def tokenize(text):
    '''Function to process text data. It preforms the following steps:
        - Find any URL in text and replace it with "urlplaceholder".
        - Remove punctuations
        - Split text into words
        - For each token: Remove stop words, normalize, trail white space and lemmatize
    Args:
        text: text data to process (as string)
    Returns:
        tokens: list of tokens after processing text data
    '''
    # detect URLs and replace by 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
                           
    # punctuation removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
                           
    # split into words
    tokens = word_tokenize(text)
                           
    # stop words removal, normalize and remove space
    tokens = [w.lower().strip() for w in tokens if w not in stopwords.words("english")]
                           
    # lemmatization
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
                           
    return tokens

# create a Named Entity Extractor, return True if a Named Entity is recognized, False otherwise
class NamedEntityExtractor(BaseEstimator, TransformerMixin):
    '''Custom transformer to recognize named entity in text data
    '''
    def named_entity(self, text):
        '''Method to check whether a named entity is present in text data.
           Args:
                text: text data
           Returns:
                True if a named entity is found
                False otherwise
        '''
        tree = ne_chunk(pos_tag(tokenize(text)))
        if 'NNP' in str(tree):
            return True
        return False
    
    def fit(self, x, y=None):
        '''Method to fit NamedEntityExtractor into text data.
        Args:
            x: data to fit the NamedEntityExtractor transformer in.
        Returns:
            self: NamedEntityExtractor itself.
        '''
        return self
    
    def transform(self, X):
        '''Method to transform input data by applying named_entity() method.
        Args:
            X: 2D array of text features
        Returns:
            dataframe that contains the transformed data.
        '''
        X_NNP = pd.Series(X).apply(self.named_entity)
        return pd.DataFrame(X_NNP)
    
def build_model():
    '''Function to build machine learning model by creating a pipeline,
    and by using GridSearchCV to find best parameters.
    Args:
        None
    Returns:
        model: machine learning pipeline that contains text process steps, classifier and GridSearchCV step.
    '''
    # create pipeline
    pipeline = Pipeline([
    ('features', FeatureUnion([

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('named_entity', NamedEntityExtractor())
    ])),

    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    # tune parameters
    parameters = {'features__text_pipeline__tfidf__norm': ['l1', 'l2'],
                    'clf__estimator__bootstrap': [True, False]}

    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1)
    return model
                           
def evaluate_model(model, X_test, Y_test, category_names):
    '''Function to predict and print out f1 score, precision and recall of the classification model.
    Args:
        model: machine learning model obtained from build_model() function
        X_test: 2D array of test features
        Y_test: 2D array of test target
        category_names: list of categories in the dataset
    Returns:
        None
    '''
    y_pred = model.predict(X_test)
    # loop over category_names list
    for i in range(len(category_names)):
        # print out each category name and its corresponding classification report
        print('Category: {}'.format(category_names[i]))
        print(classification_report(Y_test[:, i], y_pred[:, i]))
        print('\n')                       


def save_model(model, model_filepath):
    '''Function to save model as pickle object.
    Args:
        model: machine learning model
        model_filepath: path where model should be saved
    Returns:
        None
    '''
    joblib.dump(model, model_filepath)

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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()