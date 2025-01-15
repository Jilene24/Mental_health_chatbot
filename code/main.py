
import pandas as pd

from data_exploration import data_exploration
from data_cleaning import data_cleaning
from data_preprocessing import preprocess_text

def main():

    filepath = 'C:/Users/Jilen/PycharmProjects/Mental_health_chatbot/dataset/data.csv'
    data = pd.read_csv(filepath)
    data_exploration(data)
    data_cleaning(data)
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    print(data[['text', 'cleaned_text']].head())


if __name__ == "__main__":
    main()