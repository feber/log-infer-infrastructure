from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers import set_seed
import os
import gradio as gr
import torch

set_seed(42)

model_name = 'gpt2'
max_length = 242

# if not available then run on cpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Load a trained model and vocabulary that you have fine-tuned
#configuration = GPT2Config.from_json_file("/root/model_save/config.json")
model = GPT2LMHeadModel.from_pretrained("/root/model_save", return_dict_in_generate=True)
tokenizer = GPT2Tokenizer.from_pretrained("/root/model_save")

model = model.to(device)

questions_list = ["Used Utilities"]

def gpt2(text, ques):
    prompt = f"{text}, {ques}:\n"

    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    generated_outputs = model.generate(input_ids, 
                                      do_sample=True,   
                                      top_k=50, 
                                      max_length = max_length,
                                      top_p=0.95, 
                                      num_return_sequences=3, 
                                      output_scores=True)


    # only use id's that were generated
    # gen_sequences has shape [3, 15]
    gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]

    # let's stack the logits generated at each step to a tensor and transform
    # logits to probs
    probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)  # -> shape [3, 15, vocab_size]

    # now we need to collect the probability of the generated token
    # we need to add a dummy dim in the end to make gather work
    gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
    
    # now we can do all kinds of things with the probs

    # the probs that exactly those sequences are generated again
    # those are normally going to be very small
    unique_prob_per_sequence = gen_probs.prod(-1)

    # Get max value of prob out of three
    lst= list(enumerate(unique_prob_per_sequence))
    tp = max(enumerate(unique_prob_per_sequence), key=(lambda x: x[1]))

    #probability
    prob = tp[1].item()

    # output token 
    tok = gen_sequences[tp[0], :, None].squeeze(-1)
    pred = tokenizer.decode(tok, skip_special_tokens=True) 

    return pred, prob


inputs =  [ 
           gr.inputs.Textbox(lines=4, label="Input Text"),
           gr.inputs.Dropdown(questions_list, label="Ερώτηση") 
]
          

outputs = [ 
           gr.outputs.Textbox(label="GPT-2"),
           gr.outputs.Textbox(label="Score")
]

title = "GPT-2"
description = "demo for OpenAI GPT-2. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://openai.com/blog/better-language-models/'>Better Language Models and Their Implications</a> | <a href='https://github.com/openai/gpt-2'>Github Repo</a></p>"
examples = [
            ["cd /tmp"],
            [" echo  qazwsx123 nk8Y98qRrIJn4 nk8Y98qRrIJn4 n |passwd"],
            ["lscpu | grep Model"],
            ["cat /var/tmp/.var03522123 | head -n 1"],
            ["echo -e p@ssw0rd123\nIr1M520wkMx6\nIr1M520wkMx6|passwd|bash"],
            ["uname -m"],
            ["echo passwd"]
]

gr.Interface(gpt2, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(share=True)

print("ok!")
