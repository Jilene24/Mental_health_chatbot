import torch
from transformers import AutoTokenizer
from Bert_Finetuning import BertModel
from GenAI import generate_response

# Load the fine-tuned BERT model
def load_model():
    model = BertModel()
    model.load_model("fine_tuned_bert_model.pth")
    return model


def load_tokenizer():
    return AutoTokenizer.from_pretrained("mental/mental-bert-base")


def predict_label(model, tokenizer, text):
    encodings = tokenizer([text], truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model.model(**encodings)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label

def main():
    # Load the model and tokenizer

    model = load_model()
    tokenizer = load_tokenizer()

    print("Welcome to the Mental Health Chatbot!")
    print("Type your concern, and I will try to provide advice based on your input.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye! Take care of yourself.")
            break

        label = predict_label(model, tokenizer, user_input)

        advice = generate_response(label)
        print(f"\nChatbot: {advice}")

if __name__ == "__main__":
    main()
