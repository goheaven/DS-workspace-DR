# import libraries
import sys

import pandas as pd
from sqlalchemy import create_engine


# load messages dataset
def load_data(messages_filepath, categories_filepath):
    """

    :return: processed data frame
    :param messages_filepath: message data from figure8
    :param categories_filepath: categories data from figure8
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath, dtype='str')
    messages.head()
    messages = messages.drop_duplicates('id')

    # load categories dataset
    categories = pd.read_csv(categories_filepath, dtype='str')
    categories.head()
    categories = categories.drop_duplicates('id')

    # merge datasets
    df = pd.merge(messages, categories, how="left", on=['id'])


    return df


# Drop duplicated records
def clean_data(df):
    """

    :param df: dataframe
    :return: dadaframe drop dulplicated record
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = dict(categories.iloc[0])

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    for index in range(len(row)):
        row[index] = row[index][0:len(row[index]) - 2]
    category_colnames = row.values()

    # rename the columns of `categories`
    categories.columns = category_colnames

    # 转换类别值至数值 0 或 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[1]
        # convert column from string to numeric
        categories[column] = categories[column].astype("int")

    # drop the original categories column from `df`
    categories.related.replace(2, 1, inplace=True)
    df = df.drop(columns='categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.reset_index(drop=True)
    categories = categories.reset_index(drop=True)
    df = pd.concat([df, categories], axis=1, join='inner')
    #df.head()
    return df.drop_duplicates()


# Saving df to SQLite database
def save_data(df, database_filename):
    """

    :param df: dataframe as table
    :param database_filename: sqlite database name, replace table if exists
    """
    engine = create_engine("sqlite:///" + database_filename)

    df.to_sql("ETLResults", engine, if_exists="replace", index=False)
    df.to_csv("DisasterRespons.csv", encoding="utf_8_sig")


def main():
    print("sys argv", sys.argv)
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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
