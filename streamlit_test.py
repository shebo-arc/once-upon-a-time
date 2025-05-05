import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from tqdm import tqdm
import random
import os
from datasets import load_dataset


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Model
class TransformerStoryModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool, device=src.device)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt == 0)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        src_emb = self.positional_encoding(self.token_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.token_embedding(tgt) * math.sqrt(self.d_model))
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        return self.output_layer(output)

    def encode(self, src):
        src_emb = self.positional_encoding(self.token_embedding(src) * math.sqrt(self.d_model))
        src_padding_mask = (src == 0)
        return self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt, memory):
        tgt_emb = self.positional_encoding(self.token_embedding(tgt) * math.sqrt(self.d_model))
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        tgt_padding_mask = (tgt == 0)
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        return self.output_layer(output)

# Tokenizer
class SimpleTokenizer:
    def __init__(self, tokenization_type='word'):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.tokenization_type = tokenization_type
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        self.add_special_tokens()

    def add_special_tokens(self):
        self.word_to_idx = {self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

    def tokenize(self, text):
        return text.split() if self.tokenization_type == 'word' else list(text)

    def fit(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(self.tokenize(text))
        for token in sorted(vocab):
            if token not in self.word_to_idx:
                self.word_to_idx[token] = len(self.word_to_idx)
                self.idx_to_word[len(self.idx_to_word)] = token

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        return [self.word_to_idx.get(token, self.word_to_idx[self.unk_token]) for token in tokens]

    def decode(self, ids, skip_special_tokens=True):
        tokens = [self.idx_to_word.get(id, self.unk_token) for id in ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]]
        return ' '.join(tokens) if self.tokenization_type == 'word' else ''.join(tokens)

    def vocab_size(self):
        return len(self.word_to_idx)

# Text generation
def generate_story(model, tokenizer, prompt, max_length=100, temperature=1.0, device="cuda"):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    output_ids = input_ids.copy()
    for _ in range(max_length):
        curr_input = torch.tensor([output_ids[-min(len(output_ids), model.d_model):]], dtype=torch.long).to(device)
        memory = model.encode(curr_input)
        tgt_input = torch.tensor([[output_ids[-1]]], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model.decode(tgt_input, memory)
        logits = output[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, 1).item()
        output_ids.append(next_token_id)
        if next_token_id == tokenizer.word_to_idx[tokenizer.eos_token]:
            break
    return tokenizer.decode(output_ids)

# Cache the model to avoid reloading on every rerun
@st.cache_resource
def load_model():
    # Replace with your LLM model loading logic
    # Example: Using Hugging Face's GPT-2 for demonstration
    
    checkpoint = torch.load("final_model.pt", map_location="cpu")
    print(list(checkpoint.keys()))

    # Rebuild the tokenizer from saved state
    tokenizer = SimpleTokenizer(tokenization_type=checkpoint["tokenizer"]["tokenization_type"])
    tokenizer.word_to_idx = checkpoint["tokenizer"]["word_to_idx"]
    tokenizer.idx_to_word = checkpoint["tokenizer"]["idx_to_word"]

    device = "cpu"
    # Reinitialize the model with saved config
    model = TransformerStoryModel(
        vocab_size=checkpoint["config"]["vocab_size"],
        d_model=checkpoint["config"]["d_model"],
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer

# Load the model
model,tokenizer = load_model()

# Streamlit app layout
st.title("AI Story Generator")
st.markdown("Enter a prompt and customize your story!")

# User inputs
prompt = st.text_area("Story Prompt", "Once upon a time", height=100)
#genre = st.selectbox("Genre", ["Fantasy", "Sci-Fi", "Mystery", "Adventure", "Horror"])
max_length = st.slider("Story Length (words)", min_value=50, max_value=500, value=200)

# Button to generate story
if st.button("Generate Story"):
    with st.spinner("Generating your story..."):
        # Construct the input prompt with genre
        full_prompt = f"{prompt}"
        
        while True:
            # Get user input for text generation
            
            generated_text = generate_story(
                model,
                tokenizer,
                full_prompt,
                max_length=max_length,
                temperature=0.8,
                device="cpu"
            )

            print(f"Generated text:\n{generated_text}")
            break
        
        # Display the story
        st.subheader("Generated Story")
        st.write(generated_text)

# Optional: Add a clear button to reset inputs
if st.button("Clear"):
    st.session_state.prompt = ""
    st.session_state.max_length = max_length
    #st.experimental_rerun()
