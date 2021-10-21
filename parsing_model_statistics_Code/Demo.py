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

questions_list = [
                  "Αριθμός Ταυτότητας", 
                  "Ημερονηνία Έκδοσης", 
                  "Όνομα", 
                  "Given Name", 
                  "Επώνυμο", 
                  "Surname",
                  "Πατρώνυμο",
                  "Μητέρας Όνομα",
                  "Επίθετο Μητέρας",
                  "Ημερονηνία Γέννησης",
                  "Καταγωγή", 
                  "Δημότης"
]

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
            ["Y5015362 03/08/2018 Τρυφωνία Truphonia Καρκανάκη Karkanake Θέμης Φιλαρέτη Ζαχαρίου 16/07/1954 Λασσίθι Άρτα Κόρινθος 34927/4"],
            ["PO4803172 29/12/2007 Βαλέριος Balerios Θεολόγος Theologos Αχιλλέας Παντούλα Παυλή 13/08/1980 Χίος Πολύγυρος Κόρινθος 82397/5"],
            ["P7752142 21/03/2007 Δράκων Drakon Θεοδωρικάκος Theodorikakos Αριστοφάνης Ροδόκλεια Τζιόβα 15/02/1923 Θεσπρωτία Καρπενήσι Λιβαδιά 96933/9"],
            ["I6641241 06/03/2016 Μαρκέλλα Markella Κωτσιονοπούλου Kotsionopoulou Ιάσονας Κλαίρη Κοντού 10/07/1972 Πέλλα Χίος Ηγουμενίτσα 57776/4"],
            ["TH3925465 31/08/2019 Γραμματική Grammatike Κάκκα Kakka Λεμονής Μιχαέλα Ταφραλή 29/04/1935 Αργολίδα Έδεσσα Λάρισα 31165/2"],
            ["AM2456789 02/08/2018 Έριον Erion Τσάνι Tsani Μπέντρι Ζελιχά Ζετά 03/02/1995 Λούσνιε Αλβανίας Λεβαδέων 27097/1"],
            ["XY5338112 01/08/2015 Στεργιανή Stergiane Καρακώστα Karakosta Μάριος Ρωξάνη Καλαμάρα 12/02/1974 Λέσβος Ναύπλιο Λιβαδιά 41874/6"],
            ["XE5338112 06/07/2016 Ευθύμιος Efthymios Τσέργας Tsergas Κωσταντίνος Πηνελόπη Κάσσου 26/07/1987 Αθήνα Αττικής Δημητρίου 15257/6"]
]

gr.Interface(gpt2, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(share=True)

print("ok!")
