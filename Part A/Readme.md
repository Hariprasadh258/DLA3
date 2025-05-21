# Tamil Transliteration Model (Dakshina Dataset)

This repository contains a sequence-to-sequence model for transliterating Latin-script Tamil words into native Tamil script using PyTorch. It supports:

- Encoder-Decoder architectures with LSTM, GRU, or RNN
- Bidirectional encoders
- WandB for experiment tracking
- Beam search decoding

---

## Dataset

This project uses the [Dakshina dataset](https://github.com/google-research-datasets/dakshina) for training and evaluation.  


---

## How to Train

### Manual Training

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
  --bidirectional True \
  --teacher_forcing_ratio 0.3 \
  --weight_decay 1e-6
```

## Argument List

| Argument                 | Description                                      | Default |
|--------------------------|--------------------------------------------------|---------|
| --embed_dim            | Embedding dimension for encoder/decoder         | 256     |
| --hidden_dim           | Hidden size of RNN units                        | 256     |
| --rnn_type             | Type of RNN cell (rnn, lstm, gru)         | 'gru'   |
| --encoder_layers       | Number of layers in the encoder                 | 2       |
| --decoder_layers       | Number of layers in the decoder                 | 1       |
| --dropout              | Dropout rate (0 to 1)                           | 0.3     |
| --learning_rate        | Learning rate for optimizer                     | 0.001   |
| --batch_size           | Batch size for training                         | 64      |
| --epochs               | Number of training epochs                       | 10      |
| --beam_size            | Beam width for beam search decoding             | 3       |
| --bidirectional        | Use bidirectional encoder                       | True    |
| --teacher_forcing_ratio| Ratio for teacher forcing during training       | 0.3     |
| --weight_decay         | L2 regularization strength                      | 1e-5    |
| --sweep                | Enable to run WandB hyperparameter sweep        | False   |
