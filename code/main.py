
import pandas as pd

from data_exploration import data_exploration
from data_cleaning import data_cleaning
from data_preprocessing import preprocess_text
from transformers import AutoTokenizer
from Bert_Finetuning import BertModel
from sklearn.model_selection import train_test_split


def main():

    filepath = 'C:/Users/Jilen/PycharmProjects/Mental_health_chatbot/dataset/data.csv'
    data = pd.read_csv(filepath)
    data_exploration(data)
    data_cleaning(data)
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    print(data[['text', 'cleaned_text']].head())


    tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base")


    train_texts, val_texts, train_labels, val_labels = train_test_split(
        list(data['cleaned_text']),
        data['target'].values,
        test_size=0.2,
        random_state=42
    )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)


    model = BertModel()


    model.fit(train_encodings, train_labels, batch_size=8, epochs=3)




if __name__ == "__main__":
    main()