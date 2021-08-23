from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers import set_seed
import joblib
import pandas as pd
import torch

# set specific seed for a reproducible behaviour
set_seed(42)

max_length = 242

# run on GPU with CPU as fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load a trained model and vocabulary that have been fine-tuned
configuration = GPT2Config.from_json_file("/model/config.json")
tokenizer = GPT2Tokenizer.from_pretrained("/model")
model = GPT2LMHeadModel.from_pretrained("/model", config=configuration)
model = model.to(device)
model.eval()


def load_model(path: str):
    global configuration, tokenizer, model

    configuration = GPT2Config.from_json_file("/model/config.json")
    tokenizer = GPT2Tokenizer.from_pretrained("/model")
    model = GPT2LMHeadModel.from_pretrained("/model", config=configuration)
    model = model.to(device)
    model.eval()


def get_prediction(data: dict):
    """
    A function that reshapes the incoming JSON data, loads the saved model objects
    and returns the predicted class and probability.

    :param data: Dict with keys representing features and values representing the associated value
    :return: Dict with keys 'predicted_class' (class predicted) and 'predicted_prob' (probability of prediction)
    """

    # TODO: what is text, what is question?
    # prompt = f"{text}, {ques}:"
    prompt = "CMD:, cd:"

    # TODO: document the code below
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    input_len = len(generated[0])
    generated = generated.to(device)

    prediction = model.generate(
        generated,
        do_sample=True,
        top_k=50,
        max_length=max_length,
        top_p=0.95,
        num_return_sequences=1,
    )

    # TODO: document the code below
    return tokenizer.decode(prediction[0][input_len:], skip_special_tokens=True)
