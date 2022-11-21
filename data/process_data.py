import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads and merges the data to one dataframe.

    Keyword arguments:
    messages_filepath -- the file path of the message csv file
    categories_filepath -- the file path of the category csv file
    """
    #read csv data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge dataframes
    df = pd.merge(messages.drop_duplicates(),categories,on=["id"])
    #split category values into separate columns
    categories = categories["categories"].str.split(";",expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda name: name[:-2])
    categories.columns = category_colnames
    #convert values to numeric type
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda val: val[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    #merge transformed categories with whole dataframe
    df = df.drop(["categories"],axis=1)
    df = pd.concat([df,categories],axis=1)

    return df


def clean_data(df):
    """Cleans the dataframe for the machine learning model.

    Keyword arguments:
    df -- the data frame to be cleaned
    """
    #remove duplicates
    df = df.drop_duplicates()
    #remove useless column with only one value
    df = df.drop(["child_alone"],axis=1)
    #remove alls rows with values of 2 for related column from the dataframe
    df = df[df.related != 2]
    return df


def save_data(df, database_filename):
     """Saves the dataframe into a database.

    Keyword arguments:
    df -- the data frame to be stored
    database_filename -- the database to store the dataframe
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponseTable', engine, index=False)


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