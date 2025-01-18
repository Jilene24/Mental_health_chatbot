import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from Bert_Finetuning import BertModel

def preprocess_text(text):
    # Implement your text preprocessing logic here
    return text  # Example, replace with actual preprocessing

def main():
    # Load your dataset
    filepath = 'C:/Users/Jilen/PycharmProjects/Mental_health_chatbot/dataset/data.csv'
    data = pd.read_csv(filepath)

    # Preprocess and clean the data (your preprocessing steps go here)
    data['cleaned_text'] = data['text'].apply(preprocess_text)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        list(data['cleaned_text']),
        data['target'].values,
        test_size=0.2,
        random_state=42
    )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    # Custom dataset class
    class CustomDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}, torch.tensor(self.labels[idx])

    train_dataset = CustomDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = CustomDataset(val_encodings, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = BertModel()
    model.fine_tune(train_loader, epochs=3)

    # Evaluate the model
    accuracy = model.evaluate(val_loader)
    print(f"Validation Accuracy: {accuracy}")

    # Save the fine-tuned model
    model.save_model("fine_tuned_bert_model.pth")


if __name__ == "__main__":
    main()
