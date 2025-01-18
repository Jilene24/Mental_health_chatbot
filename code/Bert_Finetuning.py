import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AdamW

class BertModel:
    def __init__(self, model_name="mental/mental-bert-base", num_labels=5):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            self.optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
        return total_loss

    def evaluate(self, test_loader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                _, predicted = torch.max(logits, dim=1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        accuracy = total_correct / total_samples
        return accuracy

    def fit(self, train_encodings, train_labels, batch_size=8, epochs=3):

        train_labels = torch.tensor(train_labels)

        class CustomDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {key: torch.tensor(val[idx]) for key, val in self.encodings[idx].items()}, self.labels[idx]

        train_dataset = CustomDataset(train_encodings, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            train_loss = self.train(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss}")

    def predict(self, val_encodings):
        self.model.eval()

        input_ids = torch.tensor([encoding['input_ids'] for encoding in val_encodings])
        attention_mask = torch.tensor([encoding['attention_mask'] for encoding in val_encodings])

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted = torch.argmax(logits, dim=1)
            torch.save(self.model.state_dict(), "bert_model.pth")

        return predicted



