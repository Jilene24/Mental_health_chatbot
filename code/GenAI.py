
import openai

openai.api_key = ''

def generate_response(label):

    responses = {
        0: "Stress",
        1: "Depression",
        2: "Bipolar disorder",
        3: "Personality disorder",
        4: "Anxiety"
    }

    response = responses.get(label, "Sorry, I couldn't identify the feeling accurately.")

    if label in responses:
        gpt_response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Provide a helpful and empathetic response for a person having: {responses[label]}",
            max_tokens=100
        )
        generated_advice = gpt_response.choices[0].text.strip()
        return generated_advice
    else:
        return response

