
# Sentiment Analysis with Prefix Tuning

This project performs sentiment analysis on tweets using a T5-based model with Prefix Tuning. It is trained on the MTEB tweet sentiment extraction dataset and tracked with Weights & Biases.


## Requirements

Python 3.7+

PyTorch

Hugging Face Transformers

Datasets library

PEFT library

Weights & Biases (optional)


```bash
pip install -r requirements.txt
```


## Usage

```bash
python prefix_tune.py  --modelfile <path_to_save_model> --epochs 10 --batchsize 32 --lr 5e-4
```

* --modelfile: Path where the model will be saved after training.

* --epochs: Number of epochs for training (default: 10).

* --batchsize: Batch size for training (default: 32).

* --lr: Learning rate for the optimizer (default: 5e-4).

* --virtualtokens: Number of virtual tokens for Prefix Tuning (default: 50).

* --prefixprojection: Whether to use prefix projection (default: True).

After training, you can evaluate the model's performance by running the following:

```bash
python prefix_tune.py  --modelfile <path_to_pretrained_model> --evaluate
```












