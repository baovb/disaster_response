import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('InsertTableName', con = engine)  
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    return X, Y, Y.columns



stop_words = stopwords.words("english")

def tokenize(text):
    text = re.sub(r'[^\w\s]',' ', text.lower())
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens if w not in stop_words]
    return tokens


def build_model():
    pipeline = Pipeline([
    ('vct', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('moc', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'moc__estimator__min_samples_split': [0.1,.2,.3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred_matchIndex = pd.DataFrame(Y_pred, columns = Y_test.columns)
    for col in category_names:
        print(col)
        print(classification_report(Y_test[col], Y_pred_matchIndex[col]))


def save_model(model, model_filepath):
    import pickle
    pickle.dump(model, open(model_filepath, 'wb'))


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