
import pandas as pd

from data_exploration import data_exploration
from data_cleaning import data_cleaning

def main():

    filepath = 'C:/Users/Jilen/PycharmProjects/Mental_health_chatbot/dataset/data.csv'
    data = pd.read_csv(filepath)
    data_exploration(data)
    data_cleaning(data)



if __name__ == "__main__":
    main()