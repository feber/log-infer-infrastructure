from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers import set_seed
from typing import Union
import torch

# set specific seed for a reproducible behaviour
set_seed(42)

max_length = 242

# run on GPU with CPU as fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"this program runs on {device}")

# load a trained model and vocabulary that have been fine-tuned
# configuration: GPT2Config = GPT2Config.from_json_file("../../model/config.json")
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("../../model")
model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(
    "../../model", return_dict_in_generate=True
)
model = model.to(device)
model.eval()


def get_prediction(line: str) -> Union[str, float]:
    """
    Returns a predicted string based on a given string.
    """

    question = "Used Utilities"
    prompt = f"{line}, {question}:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    generated_outputs = model.generate(
        input_ids,
        do_sample=True,
        top_k=50,
        max_length=max_length,
        top_p=0.95,
        num_return_sequences=3,
        output_scores=True,
    )

    # only use id's that were generated
    # gen_sequences has shape [3, 15]
    gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1] :]

    # let's stack the logits generated at each step to a tensor and transform
    # logits to probs
    probs = torch.stack(generated_outputs.scores, dim=1).softmax(
        -1
    )  # -> shape [3, 15, vocab_size]

    # now we need to collect the probability of the generated token
    # we need to add a dummy dim in the end to make gather work
    gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    # now we can do all kinds of things with the probs

    # the probs that exactly those sequences are generated again
    # those are normally going to be very small
    unique_prob_per_sequence = gen_probs.prod(-1)

    # Get max value of prob out of three
    lst = list(enumerate(unique_prob_per_sequence))
    tp = max(enumerate(unique_prob_per_sequence), key=(lambda x: x[1]))

    # probability
    prob = tp[1].item()

    # output token
    tok = gen_sequences[tp[0], :, None].squeeze(-1)
    pred = tokenizer.decode(tok, skip_special_tokens=True)

    return pred, prob
