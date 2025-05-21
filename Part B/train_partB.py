import torch  # Import PyTorch for building and training neural networks
import torch.nn as nn  # Import PyTorch's neural network module
import os  # For interacting with the operating system
import wandb  # For experiment tracking using Weights & Biases
from torch.utils.data import Dataset, DataLoader  # For handling custom datasets and data loaders
from collections import defaultdict  # Dictionary subclass that provides default values
import pandas as pd  # For data manipulation using DataFrames

# Try logging into Weights & Biases using a given API key
try:
    wandb.login(key="999fe4f321204bd8f10135f3e40de296c23050f9")
except:
    print("WandB login failed - results will not be logged. Set WANDB_API_KEY in your environment.")

# Define the Encoder class, which inherits from nn.Module
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, rnn_type, dropout=0.5, bidirectional=False):
        super(Encoder, self).__init__()

        # Embedding layer to convert token indices to dense vectors
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # Save configuration parameters
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)

        # Set direction count (1 for unidirectional, 2 for bidirectional)
        self.directions = 2 if bidirectional else 1

        # Initialize the RNN layer (LSTM, GRU, or simple RNN)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0,
                               bidirectional=bidirectional)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0,
                              bidirectional=bidirectional)
        else:  # Default to RNN
            self.rnn = nn.RNN(emb_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0,
                              bidirectional=bidirectional)

        # Linear projection layer to reduce bidirectional outputs back to hidden_dim size
        if bidirectional:
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        # Apply embedding and dropout
        embedded = self.dropout(self.embedding(src))

        # If using LSTM
        if self.rnn_type == 'lstm':
            outputs, (hidden, cell) = self.rnn(embedded)

            if self.bidirectional:
                # Reshape hidden state: [num_layers, directions, batch, hidden_dim]
                hidden = hidden.view(self.num_layers, self.directions, -1, self.hidden_dim)
                # Concatenate forward and backward hidden states
                hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
                # Project to hidden_dim
                hidden = self.projection(hidden)

                # Same process for cell state
                cell = cell.view(self.num_layers, self.directions, -1, self.hidden_dim)
                cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
                cell = self.projection(cell)

                # Reshape and project outputs
                batch_size = outputs.size(0)
                seq_len = outputs.size(1)
                outputs = outputs.contiguous().view(batch_size, seq_len, self.hidden_dim * 2)
                outputs = self.projection(outputs)

                return outputs, (hidden, cell)

            return outputs, (hidden, cell)

        else:
            # For GRU or RNN
            outputs, hidden = self.rnn(embedded)

            if self.bidirectional:
                # Reshape and concatenate hidden states
                hidden = hidden.view(self.num_layers, self.directions, -1, self.hidden_dim)
                hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
                hidden = self.projection(hidden)

                # Reshape and project outputs
                batch_size = outputs.size(0)
                seq_len = outputs.size(1)
                outputs = outputs.contiguous().view(batch_size, seq_len, self.hidden_dim * 2)
                outputs = self.projection(outputs)

            return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, rnn_type, dropout=0.5, attention=False):
        super(Decoder, self).__init__()  # Initialize the parent nn.Module class
        self.embedding = nn.Embedding(output_dim, emb_dim)  # Embedding layer for target tokens
        self.rnn_type = rnn_type  # RNN type: 'lstm', 'gru', or 'rnn'
        self.num_layers = num_layers  # Number of RNN layers
        self.hidden_dim = hidden_dim  # Hidden state dimension
        self.output_dim = output_dim  # Output vocabulary size
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization
        self.attention = attention  # Boolean: use attention or not

        # Set RNN input size (add context vector size if attention is used)
        rnn_input_size = emb_dim + hidden_dim if attention else emb_dim

        # Define RNN layer based on rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(rnn_input_size, hidden_dim, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(rnn_input_size, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
        else:  # Basic RNN
            self.rnn = nn.RNN(rnn_input_size, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)

        # Define attention layers if attention is enabled
        if attention:
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)  # Linear layer for computing attention energy
            self.v = nn.Linear(hidden_dim, 1, bias=False)  # Final attention score layer

        self.fc_out = nn.Linear(hidden_dim, output_dim)  # Final output layer to generate token scores

    def forward(self, input_char, hidden, encoder_outputs=None):
        input_char = input_char.unsqueeze(1)  # Add time dimension: [batch_size] → [batch_size, 1]
        embedded = self.dropout(self.embedding(input_char))  # Embed input token: [batch_size, 1, emb_dim]

        # Apply attention if enabled
        if self.attention and encoder_outputs is not None:
            # Extract query vector from hidden state (last layer)
            if self.rnn_type == 'lstm':
                query = hidden[0][-1].unsqueeze(1)  # LSTM hidden state
            else:
                query = hidden[-1].unsqueeze(1)  # GRU/RNN hidden state

            batch_size = encoder_outputs.size(0)  # Number of examples in batch
            src_len = encoder_outputs.size(1)  # Length of encoder input sequence

            query = query.repeat(1, src_len, 1)  # Repeat query to match encoder outputs: [batch, src_len, hidden]

            # Concatenate query with encoder outputs
            energy_input = torch.cat((query, encoder_outputs), dim=2)  # [batch, src_len, 2*hidden_dim]

            # Compute attention energy scores
            energy = torch.tanh(self.attn(energy_input))  # [batch, src_len, hidden_dim]
            attention = self.v(energy).squeeze(2)  # [batch, src_len]

            # Normalize with softmax to get attention weights
            attention_weights = torch.softmax(attention, dim=1).unsqueeze(1)  # [batch, 1, src_len]

            # Compute context vector as weighted sum of encoder outputs
            context = torch.bmm(attention_weights, encoder_outputs)  # [batch, 1, hidden_dim]

            # Concatenate context with embedded input token
            rnn_input = torch.cat((embedded, context), dim=2)  # [batch, 1, emb_dim + hidden_dim]
        else:
            rnn_input = embedded  # No attention → input is just embedding

        # Pass input through RNN
        if self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(rnn_input, hidden)  # Get output and new hidden state
            hidden_state = (hidden, cell)  # Keep both for LSTM
        else:
            output, hidden = self.rnn(rnn_input, hidden)  # GRU/RNN output
            hidden_state = hidden  # Only hidden state needed

        # Generate prediction from RNN output
        prediction = self.fc_out(output.squeeze(1))  # Remove time dim: [batch, output_dim]

        # Return prediction, updated hidden state, and attention weights (if used)
        if self.attention and encoder_outputs is not None:
            return prediction, hidden_state, attention_weights.squeeze(1)  # [batch, src_len]
        else:
            return prediction, hidden_state, None  # No attention weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, rnn_type, device, use_attention=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rnn_type = rnn_type
        self.device = device
        self.use_attention = use_attention

    def forward(self, src, trg, teacher_forcing_ratio=0.0):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # For collecting attention weights
        all_attention_weights = [] if self.use_attention else None

        # Encode the source sequence
        encoder_outputs, hidden = self._encode(src)

        # Use the first token as input to start decoding
        input_char = trg[:, 0]  # <sos> token

        for t in range(1, trg_len):
            # Generate output from decoder
            if self.use_attention:
                output, hidden, attn_weights  = self.decoder(input_char, hidden, encoder_outputs)
                all_attention_weights.append(attn_weights)  # <-- ADD THIS LINE
            else:
                output, hidden, _ = self.decoder(input_char, hidden)

            outputs[:, t] = output

            # Teacher forcing: use real target or predicted token
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_char = trg[:, t] if teacher_force else top1

        return outputs, all_attention_weights

    def _encode(self, src):
        # Get encoder outputs and final hidden state
        encoder_outputs, hidden = self.encoder(src)

        # Adjust hidden state dimensions if encoder and decoder have different layers
        encoder_layers = self.encoder.num_layers
        decoder_layers = self.decoder.num_layers

        if self.rnn_type == 'lstm':
            hidden_state, cell_state = hidden

            # If encoder has fewer layers than decoder, pad with zeros
            if encoder_layers < decoder_layers:
                padding = torch.zeros(
                    decoder_layers - encoder_layers,
                    hidden_state.size(1),
                    hidden_state.size(2)
                ).to(self.device)
                hidden_state = torch.cat([hidden_state, padding], dim=0)
                cell_state = torch.cat([cell_state, padding], dim=0)

            # If encoder has more layers than decoder, truncate
            elif encoder_layers > decoder_layers:
                hidden_state = hidden_state[:decoder_layers]
                cell_state = cell_state[:decoder_layers]

            # Make sure hidden dimensions match decoder's expected dimensions
            if hidden_state.size(2) != self.decoder.hidden_dim:
                # Project hidden state to the decoder's dimension using a linear projection
                batch_size = hidden_state.size(1)
                proj_hidden = torch.zeros(
                    hidden_state.size(0),
                    batch_size,
                    self.decoder.hidden_dim
                ).to(self.device)

                for layer in range(hidden_state.size(0)):
                    # Simple linear projection for each layer
                    proj_hidden[layer] = hidden_state[layer].clone().view(batch_size, -1)[:, :self.decoder.hidden_dim]

                # Apply the same projection to cell state
                proj_cell = torch.zeros_like(proj_hidden)
                for layer in range(cell_state.size(0)):
                    proj_cell[layer] = cell_state[layer].clone().view(batch_size, -1)[:, :self.decoder.hidden_dim]

                hidden = (proj_hidden, proj_cell)
            else:
                hidden = (hidden_state, cell_state)
        else:
            # For GRU and RNN
            # If encoder has fewer layers than decoder, pad with zeros
            if encoder_layers < decoder_layers:
                padding = torch.zeros(
                    decoder_layers - encoder_layers,
                    hidden.size(1),
                    hidden.size(2)
                ).to(self.device)
                hidden = torch.cat([hidden, padding], dim=0)
            # If encoder has more layers than decoder, truncate
            elif encoder_layers > decoder_layers:
                hidden = hidden[:decoder_layers]

            # Make sure hidden dimensions match decoder's expected dimensions
            if hidden.size(2) != self.decoder.hidden_dim:
                # Project hidden state to the decoder's dimension
                batch_size = hidden.size(1)
                proj_hidden = torch.zeros(
                    hidden.size(0),
                    batch_size,
                    self.decoder.hidden_dim
                ).to(self.device)

                for layer in range(hidden.size(0)):
                    # Simple linear projection for each layer
                    proj_hidden[layer] = hidden[layer].clone().view(batch_size, -1)[:, :self.decoder.hidden_dim]

                hidden = proj_hidden

        return encoder_outputs, hidden

# Character-level vocabulary builder
def build_vocab(tokens):
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for token in tokens:
        for char in token:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab

def encode_sequence(seq, vocab):
    return [vocab.get(char, vocab['<unk>']) for char in seq]

class DakshinaDataset(Dataset):
    def __init__(self, data_path, latin_vocab=None, devanagari_vocab=None):
        self.latin_words = []
        self.devanagari_words = []

        # Group all transliterations by Devanagari word
        candidates = defaultdict(list)

        with open(data_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                native, latin, rel = parts[0], parts[1], int(parts[2])
                candidates[native].append((latin, rel))

        # Keep only the transliteration(s) with highest score for each native word
        for native, translits in candidates.items():
            max_rel = max(rel for _, rel in translits)
            for latin, rel in translits:
                if rel == max_rel:
                    self.latin_words.append(latin)
                    self.devanagari_words.append(native)

        print(f"Dataset from {data_path}: {len(self.latin_words)} pairs.")

        self.latin_vocab = latin_vocab or build_vocab(self.latin_words)
        self.devanagari_vocab = devanagari_vocab or build_vocab(self.devanagari_words)

    def __len__(self):
        return len(self.latin_words)

    def __getitem__(self, idx):
        src_seq = encode_sequence(self.latin_words[idx], self.latin_vocab)
        trg_seq = encode_sequence(self.devanagari_words[idx], self.devanagari_vocab)
        # add <sos> and <eos> tokens for target sequences
        trg_seq = [self.devanagari_vocab['<sos>']] + trg_seq + [self.devanagari_vocab['<eos>']]
        return src_seq, trg_seq

def collate_fn(batch):
    # batch is a list of tuples (src_seq, trg_seq)
    src_seqs, trg_seqs = zip(*batch)

    # find max lengths
    max_src_len = max(len(seq) for seq in src_seqs)
    max_trg_len = max(len(seq) for seq in trg_seqs)

    # pad sequences
    src_padded = [seq + [0]*(max_src_len - len(seq)) for seq in src_seqs]
    trg_padded = [seq + [0]*(max_trg_len - len(seq)) for seq in trg_seqs]

    # convert to tensors
    src_tensor = torch.tensor(src_padded, dtype=torch.long)
    trg_tensor = torch.tensor(trg_padded, dtype=torch.long)

    return src_tensor, trg_tensor

def compute_word_accuracy(preds, trg, pad_idx, sos_idx=1, eos_idx=2):
    """
    Compute word-level accuracy: a word is correct only if all tokens match (excluding pad, sos, eos).
    preds, trg: [batch_size, seq_len]
    """
    batch_size = preds.size(0)
    correct = 0

    for i in range(batch_size):
        # Get sequence for this example (exclude pad, sos, eos tokens)
        pred_seq = [idx for idx in preds[i].tolist() if idx not in [pad_idx, sos_idx, eos_idx]]
        trg_seq = [idx for idx in trg[i].tolist() if idx not in [pad_idx, sos_idx, eos_idx]]

        # Compare full sequences (exact match)
        if pred_seq == trg_seq:
            correct += 1

    return correct, batch_size

def beam_search(model, src_seq, src_vocab, tgt_vocab, beam_width=3, max_len=20):
    model.eval()
    index_to_char = {v: k for k, v in tgt_vocab.items()}
    device = model.device

    # Prepare input
    src_indices = encode_sequence(src_seq, src_vocab)
    src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)

    # Get encoder outputs and hidden state
    encoder_outputs, hidden = model._encode(src_tensor)

    # Start with start-of-sequence token
    beams = [([tgt_vocab['<sos>']], 0.0, hidden)]

    for _ in range(max_len):
        new_beams = []
        # for seq, score, hidden in beams:
        #     last_token = torch.tensor([seq[-1]], dtype=torch.long).to(device)

        #     # Use attention if model has it
        #     if model.use_attention:
        #         output, new_hidden, _ = model.decoder(last_token, hidden, encoder_outputs)
        #     else:
        #         output, new_hidden = model.decoder(last_token, hidden)

        #     log_probs = torch.log_softmax(output, dim=-1)
        #     topk = torch.topk(log_probs, beam_width)

        #     for prob, idx in zip(topk.values[0], topk.indices[0]):
        #         new_seq = seq + [idx.item()]
        #         new_score = score + prob.item()
        #         new_beams.append((new_seq, new_score, new_hidden))
        for seq, score, hidden in beams:
            if seq[-1] == tgt_vocab['<eos>']:
                # Don't extend this beam; just carry it forward
                new_beams.append((seq, score, hidden))
                continue
        
            last_token = torch.tensor([seq[-1]], dtype=torch.long).to(device)
        
            # Use attention if model has it
            if model.use_attention:
                output, new_hidden, _ = model.decoder(last_token, hidden, encoder_outputs)
            else:
                output, new_hidden = model.decoder(last_token, hidden)
        
            log_probs = torch.log_softmax(output, dim=-1)
            topk = torch.topk(log_probs, beam_width)
        
            for prob, idx in zip(topk.values[0], topk.indices[0]):
                new_seq = seq + [idx.item()]
                new_score = score + prob.item()
                new_beams.append((new_seq, new_score, new_hidden))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        # Stop if all beams end with EOS
        if all(seq[-1] == tgt_vocab['<eos>'] for seq, _, _ in beams):
            break

    # Pick the best beam
    best_seq = beams[0][0]
    # Remove special tokens for output
    decoded = [index_to_char[i] for i in best_seq if i not in {tgt_vocab['<sos>'], tgt_vocab['<eos>'], tgt_vocab['<pad>']}]
    return ''.join(decoded)

def train(model, dataloader, optimizer, criterion, clip=1, teacher_forcing_ratio=0.0):
    model.train()
    epoch_loss = 0
    total_words = 0
    correct_words = 0

    pad_idx = 0  # Pad index in vocabulary
    sos_idx = 1  # Start of sequence index
    eos_idx = 2  # End of sequence index

    for src, trg in dataloader:
        src, trg = src.to(model.device), trg.to(model.device)
        optimizer.zero_grad()

        # Generate sequence
        output, _ = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        output_dim = output.shape[-1]

        # Ignore first token (<sos>) in loss calculation
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg_flat = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg_flat)
        loss.backward()

        # Use gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        # Calculate word accuracy
        pred_tokens = output.argmax(1).view(trg[:, 1:].shape)  # [batch_size, trg_len-1]
        trg_trimmed = trg[:, 1:]                             # [batch_size, trg_len-1]

        correct, total = compute_word_accuracy(pred_tokens, trg_trimmed, pad_idx, sos_idx, eos_idx)
        correct_words += correct
        total_words += total

    avg_loss = epoch_loss / len(dataloader)
    word_acc = correct_words / total_words if total_words > 0 else 0

    return avg_loss, word_acc * 100

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    total_words = 0
    correct_words = 0

    pad_idx = 0  # Pad index in vocabulary
    sos_idx = 1  # Start of sequence index
    eos_idx = 2  # End of sequence index

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(model.device), trg.to(model.device)

            # Generate full sequence with no teacher forcing
            output, _  = model(src, trg, teacher_forcing_ratio=0.0)
            # Visualize for the first example in batch
            # attn = torch.stack([aw[0] for aw in attention_weights]).cpu().numpy()
            output_dim = output.shape[-1]

            # Ignore first token (<sos>) in loss calculation
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg_flat = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg_flat)
            epoch_loss += loss.item()

            # Calculate word accuracy
            pred_tokens = output.argmax(1).view(trg[:, 1:].shape)
            trg_trimmed = trg[:, 1:]

            correct, total = compute_word_accuracy(pred_tokens, trg_trimmed, pad_idx, sos_idx, eos_idx)
            correct_words += correct
            total_words += total

    avg_loss = epoch_loss / len(dataloader)
    word_acc = correct_words / total_words if total_words > 0 else 0

    return avg_loss, word_acc * 100

def predict_examples(model, dataloader, latin_index_to_token, devanagari_index_to_token, n=4229):
    """Show a few examples of model predictions vs actual targets"""
    model.eval()
    pad_idx = 0
    sos_idx = 1  # Start of sequence
    eos_idx = 2  # End of sequence
    count = 0
    results = []

    print("\nPrediction Examples:")
    print("-" * 60)

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(model.device), trg.to(model.device)
            output, _ = model(src, trg, teacher_forcing_ratio=0.0)
            # print(len(output))
            pred_tokens = output.argmax(-1)  # [batch_size, seq_len]

            for i in range(min(src.size(0), n - count)):
                # Decode input
                input_indices = [idx for idx in src[i].tolist() if idx != pad_idx]
                input_tokens = [latin_index_to_token.get(idx, '<unk>') for idx in input_indices]
                input_text = "".join(input_tokens)

                # Decode target
                target_indices = [idx for idx in trg[i].tolist() if idx not in [pad_idx, sos_idx, eos_idx]]
                target_tokens = [devanagari_index_to_token.get(idx, '<unk>') for idx in target_indices]
                target_text = "".join(target_tokens)

                # Decode prediction
                pred_indices = [idx for idx in pred_tokens[i].tolist() if idx not in [pad_idx, sos_idx, eos_idx]]
                pred_tokens_text = [devanagari_index_to_token.get(idx, '<unk>') for idx in pred_indices]
                pred_text = "".join(pred_tokens_text)

                result = {
                    "input": input_text,
                    "target": target_text,
                    "prediction": pred_text,
                    "correct": pred_text == target_text
                }
                results.append(result)

                # print(f"Input:     {input_text}")
                # print(f"Target:    {target_text}")
                # print(f"Predicted: {pred_text}")
                # print("-" * 60)

                count += 1
                if count >= n:
                    break
            if count >= n:
                break

    return results

import matplotlib.pyplot as plt
import seaborn as sns

# Define sweep configuration with improved parameters
def get_sweep_config():
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'},
            'parameters': {
            'embed_dim': {'values': [384]},
            'hidden_dim': {'values': [256]},
            'rnn_type': {'values': ['gru']},
            'encoder_layers': {'values': [2]},
            'decoder_layers': {'values': [1]},
            'dropout': {'values': [0.3]},
            'learning_rate': {'values': [0.001]},
            'batch_size': {'values': [64]},
            'epochs': {'values': [5]},
            'beam_size': {'values': [3]},
            'use_attention': {'values': [True]},
            'bidirectional': {'values': [True]},
            'teacher_forcing_ratio': {'values': [0.3]},
            'weight_decay': {'values': [1e-6]}
        }
    }
    return sweep_config

# Main training function for sweep runs
def train_sweep():
    # Initialize wandb with sweep configuration
    run = wandb.init(project="transliteration-model")

    # Access hyperparameters from wandb.config
    config = run.config

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define data paths (adjust for Kaggle environment)
    # ta.translit.sampled.dev.tsv
    data_dir = '/kaggle/input/dakshina/dakshina_dataset_v1.0/ta/lexicons/'
    train_path = os.path.join(data_dir, 'ta.translit.sampled.train.tsv')
    dev_path = os.path.join(data_dir, 'ta.translit.sampled.dev.tsv')
    test_path = os.path.join(data_dir, 'ta.translit.sampled.test.tsv')

    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Could not find training data at {train_path}. Please check the path.")

    # Load datasets
    print("Loading training dataset...")
    train_dataset = DakshinaDataset(train_path)
    latin_vocab = train_dataset.latin_vocab
    devanagari_vocab = train_dataset.devanagari_vocab

    print("Loading validation dataset...")
    val_dataset = DakshinaDataset(
        dev_path,
        latin_vocab=latin_vocab,
        devanagari_vocab=devanagari_vocab
    )

    print("Loading test dataset...")
    test_dataset = DakshinaDataset(
        test_path,
        latin_vocab=latin_vocab,
        devanagari_vocab=devanagari_vocab
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Get vocabulary information
    latin_vocab_size = len(latin_vocab)
    devanagari_vocab_size = len(devanagari_vocab)
    pad_idx = devanagari_vocab['<pad>']

    # Log vocabulary sizes
    wandb.log({"latin_vocab_size": latin_vocab_size, "devanagari_vocab_size": devanagari_vocab_size})

    # Generate a model name based on hyperparameters
    model_name = f"{config.rnn_type}_ed{config.embed_dim}_hid{config.hidden_dim}_enc{config.encoder_layers}_dec{config.decoder_layers}_attn{config.use_attention}_drop{config.dropout}"
    wandb.run.name = model_name

    # Create model architecture
    encoder = Encoder(
        input_dim=latin_vocab_size,
        emb_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.encoder_layers,
        rnn_type=config.rnn_type,
        dropout=config.dropout,
        bidirectional=config.bidirectional
    )

    decoder = Decoder(
        output_dim=devanagari_vocab_size,
        emb_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.decoder_layers,
        rnn_type=config.rnn_type,
        dropout=config.dropout,
        attention=config.use_attention
    )

    model = Seq2Seq(
        encoder,
        decoder,
        rnn_type=config.rnn_type,
        device=device,
        use_attention=config.use_attention
    ).to(device)

    # # Count and log the number of model parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # wandb.log({
    #     "total_parameters": total_params,
    #     "trainable_parameters": trainable_params
    # })
    # print(f"Model: {model_name}")
    # print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")

    # Setup optimizer and loss function with weight decay for regularization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 5  # Increased patience
    patience_counter = 0

    # Save directory for models
    model_dir = '/kaggle/working'
    os.makedirs(model_dir, exist_ok=True)

    # Track metrics for each epoch
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")

        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, clip=1.0,
                                      teacher_forcing_ratio=config.teacher_forcing_ratio)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        # Save best model and check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            # Save the best model
            best_model_path = os.path.join(model_dir, f"{model_name}_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    # Load the best model for testing
    best_model_path = os.path.join(model_dir, f"{model_name}_best.pt")
    try:
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for testing")
    except:
        print("Using current model for testing (best model not found)")

    # Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Results -> Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    # Log final test metrics
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc
    })

    # Create index-to-token dictionaries for prediction display
    latin_index_to_token = {idx: token for token, idx in latin_vocab.items()}
    devanagari_index_to_token = {idx: token for token, idx in devanagari_vocab.items()}

    # Generate prediction examples for visualization
    example_results = predict_examples(
        model,
        test_loader,
        latin_index_to_token,
        devanagari_index_to_token,
        n=4992
    )

    # Log the examples as a table in wandb
    example_table = wandb.Table(
        columns=["Input", "Target", "Prediction", "Correct"]
    )
    for result in example_results:
        example_table.add_data(
            result["input"],
            result["target"],
            result["prediction"],
            result["correct"]
        )
    wandb.log({"prediction_examples": example_table})

    # Test beam search if enabled
    if config.beam_size > 1:
        print(f"\nTesting beam search with beam width {config.beam_size}...")
        beam_correct = 0
        beam_total = 0
        all_predictions=[]

        for src, trg in test_loader:
            src = src.to(device)
            trg = trg.to(device)
            for i in range(min(5, src.size(0))):  # Test beam search on a few examples
                # Get input sequence
                src_seq = [latin_index_to_token[idx] for idx in src[i].tolist() if idx != pad_idx]
                src_text = ''.join(src_seq)

                # Get target sequence
                trg_indices = [idx for idx in trg[i].tolist() if idx not in [pad_idx, 1]]  # Remove <pad> and <sos>
                trg_text = ''.join([devanagari_index_to_token.get(idx, '<unk>') for idx in trg_indices])

                # Run beam search
                beam_pred = beam_search(
                    model,
                    src_text,
                    latin_vocab,
                    devanagari_vocab,
                    beam_width=config.beam_size,
                    max_len=30
                )

                beam_correct += 1 if beam_pred == trg_text else 0
                beam_total += 1

                all_predictions.append({
                    "Input": src_text,
                    "Target": trg_text,
                    "Prediction": beam_pred,
                    "Correct": beam_pred == trg_text
                        })

                print(f"Input: {src_text}")
                print(f"Target: {trg_text}")
                print(f"Beam Pred: {beam_pred}")
                print("-" * 60)
        # Create DataFrame from all predictions
        predictions_df = pd.DataFrame(all_predictions)

        # Save to CSV
        csv_path = os.path.join(model_dir, f"{model_name}_predictions.csv")
        predictions_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        # predictions_df.to_csv(csv_path, index=False)
        print(f"\nSaved all predictions to {csv_path}")

        beam_acc = beam_correct / beam_total * 100 if beam_total > 0 else 0
        print(f"Beam search accuracy: {beam_acc:.2f}%")
        wandb.log({"beam_search_accuracy": beam_acc})

    # plot_attention_grid(model, test_loader, idx_to_src_token=latin_index_to_token, idx_to_tgt_token=devanagari_index_to_token)

    return model, latin_vocab, devanagari_vocab


import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

def plot_attention_grid(model, dataloader, idx_to_src_token=None, idx_to_tgt_token=None, num_samples=9):
    model.eval()

    # Load Tamil font (update the path as per your font location)
    tamil_font_path = "/kaggle/input/notations-tamil1/static/NotoSansTamil-Regular.ttf"  # Put this font file in the working dir or give full path
    tamil_font = font_manager.FontProperties(fname=tamil_font_path)

    samples_plotted = 0
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(model.device), trg.to(model.device)

            # Forward pass with no teacher forcing to get attention
            output, attention_weights = model(src, trg, teacher_forcing_ratio=0.0)

            if not attention_weights:
                print("No attention weights returned by the model.")
                return

            # Shape: [tgt_len-1, batch_size, src_len]
            attn_tensor = torch.stack(attention_weights)

            batch_size = src.size(0)

            for i in range(batch_size):
                if samples_plotted >= num_samples:
                    break

                # Tokens
                if idx_to_src_token and idx_to_tgt_token:
                    src_tokens = [idx_to_src_token[idx.item()] for idx in src[i]]
                    tgt_tokens = [idx_to_tgt_token[idx.item()] for idx in trg[i][1:]]  # skip <sos>
                else:
                    src_tokens = None
                    tgt_tokens = None

                # Remove <pad> tokens and track lengths
                src_tokens_clean = [tok for tok in src_tokens if tok != "<pad>"]
                tgt_tokens_clean = [tok for tok in tgt_tokens if tok != "<pad>"]
                src_trim_len = len(src_tokens_clean)
                tgt_trim_len = len(tgt_tokens_clean)

                # Trim attention to match cleaned tokens
                attn_for_sample = attn_tensor[:, i, :].cpu().numpy()  # [tgt_len-1, src_len]
                attn_for_sample = attn_for_sample[:tgt_trim_len, :src_trim_len]

                # Plot
                ax = axes[samples_plotted]
                sns.heatmap(attn_for_sample, cmap="viridis",
                            xticklabels=src_tokens_clean,
                            yticklabels=tgt_tokens_clean,
                            ax=ax)
                ax.set_xlabel("Source Tokens", fontproperties=tamil_font)
                ax.set_ylabel("Target Tokens", fontproperties=tamil_font)
                ax.set_title(f"Sample {samples_plotted + 1}", fontproperties=tamil_font)
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='y', rotation=0)

                # Apply Tamil font to tick labels
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontproperties(tamil_font)

                samples_plotted += 1

            if samples_plotted >= num_samples:
                break
    plt.savefig("heatmap.png", dpi = 300)
    # wandb.log("heatmap.png")
    wandb.log({"attention_heatmap": wandb.Image("heatmap.png")})
    plt.tight_layout()
    plt.show()


# Entry point - runs a wandb sweep
def run_wandb_sweep():
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="transliteration-model-tam1")
    wandb.agent(sweep_id, train_sweep, count=1)


# Main execution block for Kaggle
if __name__ == "__main__":
    # Run the wandb sweep
    run_wandb_sweep()