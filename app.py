from flask import Flask, request, jsonify, render_template
import random
import torch
from datetime import datetime
from torch import nn
from torch.optim import Adam
from transformers import GPT2Model, GPT2Tokenizer

class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes,max_seq_len, gpt_model_name):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)

        
    def forward(self, input_id, mask):
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model_new = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=2, max_seq_len=128, gpt_model_name="gpt2")
model_new.load_state_dict(torch.load("model2M-1epoch.pt", map_location=torch.device('cpu')))

app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"

@app.route('/')
def home():
    return render_template('home.html', prediction_text="")

@app.route('/predict', methods=['GET','POST'])
def predict():
	a_description = request.form.get('description')
	text=str(a_description)
	fixed_text = " ".join(text.lower().split())
	model_input = tokenizer(fixed_text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
	mask = model_input['attention_mask'].cpu()
	input_id = model_input["input_ids"].squeeze(1).cpu()
	output = model_new(input_id, mask)
	prob = torch.nn.functional.softmax(output, dim=1)[0]
	labels_map = {
	    0: "Non-Sexist",
	    1: "Sexist"
		 }
	pred_label = labels_map[output.argmax(dim=1).item()]
	now = datetime.now()
	date_time = now.strftime("%m/%d/%Y %H:%M:%S")
	if pred_label=="Sexist":
		out = str(date_time) + ' | ' + pred_label + '     : ' + a_description + '\n'
	else:
		out = str(date_time) + ' | ' + pred_label + ' : ' + a_description + '\n'
	#out = str(date_time) + ' | ' + pred_label + ' : ' + a_description + '\n'
	file = open("static/log.txt", "a")
	file.write(out)
	file.close()
	return render_template('home.html', prediction_text=pred_label)

if __name__ == "__main__":
    app.run("0.0.0.0",8383,threaded=True,debug=True)
