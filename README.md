# Tamil Transliteration Model (Dakshina Dataset)

This repository contains a sequence-to-sequence model for transliterating Latin-script Tamil words into native Tamil script using PyTorch. It supports:

- Encoder-Decoder architectures with LSTM, GRU, or RNN
- Optional Attention mechanism
- Bidirectional encoders
- WandB for experiment tracking
- Beam search decoding
- Attention heatmap visualization

---

## Dataset

This project uses the [Dakshina dataset](https://github.com/google-research-datasets/dakshina) for training and evaluation.  


---

## How to Train

### Option 1: Manual Training (via argparse)

```
python train_partA.py \
  --embed_dim 384 \
  --hidden_dim 256 \
  --rnn_type gru \
  --encoder_layers 2 \
  --decoder_layers 1 \
  --dropout 0.3 \
  --learning_rate 0.001 \
  --batch_size 64 \
  --epochs 5 \
  --beam_size 3 \
  --use_attention True \
  --bidirectional True \
  --teacher_forcing_ratio 0.3 \
  --weight_decay 1e-
