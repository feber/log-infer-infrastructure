from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers import set_seed
import torch

# set specific seed for a reproducible behaviour
set_seed(42)

max_length = 242

# run on GPU with CPU as fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"this program runs on {device}")

# load a trained model and vocabulary that have been fine-tuned
configuration = GPT2Config.from_json_file("../model/config.json")
tokenizer = GPT2Tokenizer.from_pretrained("../model")
model = GPT2LMHeadModel.from_pretrained("../model", config=configuration)
model = model.to(device)
model.eval()


def load_model(path: str):
    global configuration, tokenizer, model

    configuration = GPT2Config.from_json_file("../model/config.json")
    tokenizer = GPT2Tokenizer.from_pretrained("../model")
    model = GPT2LMHeadModel.from_pretrained("../model", config=configuration)
    model = model.to(device)
    model.eval()


def get_prediction(line: str):
    """
    Returns a predicted string based on a given string.
    """

    # for now, question is always `Utility`
    question = "Utility"
    prompt = f"{line}, {question}:"

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
