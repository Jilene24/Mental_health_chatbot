import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW

class BertModel:
    def __init__(self, model_name="mental/mental-bert-base", num_labels=5):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def fine_tune(self, train_loader, epochs=3):

        self.model.train()
        total_loss = 0
        for epoch in range(epochs):
            for batch in train_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

    def evaluate(self, val_loader):

        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Prediction and accuracy calculation
                _, predicted = torch.max(logits, dim=1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        return accuracy

    def save_model(self, save_path="bert_model.pth"):

        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, model_path="bert_model.pth"):

        self.model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")



