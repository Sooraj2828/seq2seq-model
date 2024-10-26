# app.py
from flask import Flask, request, jsonify, render_template
import torch
import pickle
from model import EncoderRNN, AttnDecoderRNN  # Assuming model classes are in a separate file
import torch.nn.functional as F

app = Flask(_name_)

# Load model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 250
MAX_LENGTH = 100
SOS_token = 0
EOS_token = 1
UNK_token = 2

# Load input and output language files
with open('input_lang.pkl', 'rb') as f:
    input_lang = pickle.load(f)
with open('output_lang.pkl', 'rb') as f:
    output_lang = pickle.load(f)

# Load encoder and decoder
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
decoder.load_state_dict(torch.load('decoder.pth', map_location=device))
encoder.eval()
decoder.eval()

# Helper function for processing input
def tensorFromSentence(lang, sentence):
    return torch.tensor([lang.word2index.get(word, UNK_token) for word in sentence.split(' ')] + [EOS_token], dtype=torch.long, device=device).view(-1, 1)

# Evaluation function
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()
        
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return ' '.join(decoded_words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    sentence = data.get('sentence')
    translation = evaluate(encoder, decoder, sentence)
    return jsonify({'translation': translation})

if _name_ == '_main_':
    app.run(debug=True)
