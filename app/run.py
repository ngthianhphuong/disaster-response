import json
import plotly
import pandas as pd
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin

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
    


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # DATA for My Graph 1
    # create a Series named length, that contains the word count of every message
    length = df.message.apply(lambda x: len(x))
    y = ['mean', 'median', 'stdev']
    x = [length.mean(), length.median(), length.std()]
    
    # DATA for My Graph 2
    categories = df.iloc[:, 4:].columns
    categories_count = df.iloc[:, 4:].sum(axis = 0)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # provided graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts, opacity = 0.5,
                    marker={'color':['#040d17', '#12345b', '#1f5aa0']}
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # My Graph 1
         {
            'data': [
                Bar(
                    x = x,
                    y = y,
                    text = y,
                    textposition='auto',
                    orientation="h", opacity = 0.5,
                    marker={'color':['#160203', '#580a0d', '#9a1118']}
                )
            ],

            'layout': {
                'title': 'Main Statistical Metrics of Message Length (in words)',
                'yaxis': {
                    'title': "Main Statistical Metrics"
                },
                'xaxis': {
                    'title': "Value"
                }
            }
        },
        
        # My Graph 2
         {
            'data': [
                Bar(
                    x = categories,
                    y = categories_count,
                    marker={'color':['rgb(90, 72, 164)', 'rgb(172, 40, 87)']*18}
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()