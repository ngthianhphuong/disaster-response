import sys
import pandas as pd 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Function to load and merge data from 2 input datasets that contain messages and categories.
    Args:
        messages_filepath: path to dataset that contains messages
        categories_filepath: path to dataset that contains categories
    Returns:
        df: merged dataset from messages and categories datasets
    '''
    # load data from 2 provided datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # drop duplicated rows
    messages = messages.drop_duplicates()
    categories = categories.drop_duplicates()
    
    # same IDs may correspond to different messages or categories
    # drop them to obtain unique IDs in each dataset before merging
    messages.drop_duplicates(subset='id', keep = False, inplace = True)
    categories.drop_duplicates(subset='id', keep = False, inplace = True)
    
    # merge on IDs
    df = pd.merge(messages, categories, on='id', how='inner')
    return df

def clean_data(df):
    '''Function to clean data, essentially to split categories column so that each category becomes a separate column.
    Args: 
        df: dataframe obtained from load_data() function
    Returns:
        df_new: dataframe after cleaning
    '''
    # create a new df named cats_df that contains 36 categories columns
    cats_df = df.categories.str.split(';', expand=True)
    
    # use first row of cats_df dataframe to extract a list of new column names for cats_df
    # rename the columns of cats_df
    cats_df.columns = cats_df.iloc[0,:].apply(lambda x: x[:-2])
    
    # convert category values to just numbers 0 or 1
    for column in cats_df:
        # set each value to be the last character of the string
        cats_df[column] = cats_df[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        cats_df[column] = pd.to_numeric(cats_df[column])
        
    # drop the original categories column from `df`
    df.drop(columns= 'categories', axis = 1, inplace = True)
    # concatenate the original df with the new cats_df dataframe
    df_new = pd.concat([df, cats_df], axis = 1)
    return df_new

def save_data(df, database_filename):
    '''Function to save cleaned dataframe to database.
    Args:
        df: dataframe after cleaning, obtained from clean_data() function
        database_filename: database file name. Example: MyDatabase.db
    Returns:
        None
    '''
    # save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename, engine, index=False, if_exists='replace') 
    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()