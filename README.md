# SASRec - MovieLens 1M Sequential Recommendation

This repository implements **SASRec (Self-Attentive Sequential Recommendation)** on the **MovieLens 1M** dataset.

The model learns from each user's historical interaction sequence and predicts the next most likely movie. The project supports offline evaluation with `Recall@K` and `NDCG@K`.

## Project Structure

```text
sasrec/
├── main.py           # Training / evaluation entry point
├── data.py           # Data loading, sequence building, negative sampling
├── Net.py            # SASRec model implementation
├── SasrecData.py     # Dataset and collate_fn
├── ml-1m/            # MovieLens 1M dataset
│   ├── ratings.dat
│   ├── users.dat
│   ├── movies.dat
│   └── README
├── sasrec_model.pth  # Saved model weights
└── sasrec_eval.log   # Evaluation log
```

## Dataset

The project uses the following MovieLens 1M files:

- `ratings.dat` — user-item interactions with timestamps
- `users.dat` — user profile information
- `movies.dat` — movie title, year, and genres

### Data Processing Pipeline

1. Read `ratings.dat` and build each user's interaction history.
2. Sort interactions by timestamp to obtain ordered sequences.
3. Encode movie IDs into consecutive indices, with `0` reserved for padding.
4. Generate training samples:
   - `item_seq`: historical sequence
   - `pos_item`: the next ground-truth item
   - `neg_items`: negative samples
5. Generate validation samples by keeping the last interaction for each user.

## Model Overview

`Net.py` contains a simplified SASRec implementation with the following components:

- `Embedding_layer`: item embedding layer
- `Self_Attention`: self-attention block with causal and padding masking
- `FFN`: feed-forward network
- `Sasrec_Net`: the full recommendation model

The model takes a user sequence and mask as input, produces a sequence representation, and scores positive/negative items by matching them with the user representation.

## Requirements

Recommended environment:

- Python 3.11+
- `numpy`
- `torch`
- `scikit-learn`
- `tqdm`

Install dependencies with:

```bash
pip install numpy torch scikit-learn tqdm
```

## Usage

Run the project from the `sasrec` directory:

```bash
python main.py
```

Note: `data.py` uses relative paths such as `./ml-1m/*.dat`, so the script must be executed from the `sasrec` directory.

### Command-Line Examples

By default, `main.py` runs in evaluation mode. You can override the configuration from the command line:

```bash
# Default evaluation
python main.py

# Change sequence length, negative samples, and Top-K
python main.py --max-len 100 --neg-sample-num 5 --topk 20

# Train the model with custom hyperparameters
python main.py --mode train --epoch 50 --lr 0.0005 --batch-size 64 --dropout 0.3
```

Common options:

- `--max-len`: maximum sequence length
- `--neg-sample-num`: number of negative samples for training
- `--batch-size`: batch size
- `--embedding-size`: item embedding dimension
- `--dropout`: dropout rate
- `--epoch`: number of training epochs
- `--lr`: learning rate
- `--eval-every`: evaluate every N epochs
- `--topk`: Top-K for evaluation
- `--mode`: `eval` or `train`
- `--model-path`: model save/load path
- `--log-file`: log file path

## Training and Evaluation

### Default Behavior

- Load and prepare the dataset
- Load `sasrec_model.pth` if it already exists
- Run one evaluation pass and write the result to `sasrec_eval.log`

### Training Mode

To train the model, run:

```bash
python main.py --mode train
```

Training mode will:

- Optimize the model with `BCEWithLogitsLoss`
- Evaluate every few epochs
- Save the best/current model to `sasrec_model.pth`

## Metrics

The evaluation reports:

- `Recall@K`
- `NDCG@K`

The default value of `K` is `10`.

## Output Files

- `sasrec_model.pth`: model weights
- `sasrec_eval.log`: evaluation log

## Notes

1. Always run the script from the `sasrec` directory.
2. When loading a model across devices, `map_location=device` is recommended.
3. The padding mask should be built from `seq != 0` to avoid treating padding as real history.
4. During evaluation, candidates are ranked among the target item plus sampled negatives, which is a standard offline evaluation setup.

## Future Improvements

- Make more training hyperparameters configurable through the CLI
- Add a dedicated evaluation script and visualization utilities
- Split preprocessing and training into more modular components

## Summary

This project is a MovieLens 1M sequence recommendation experiment for learning SASRec-style sequence modeling, masking, negative sampling, and offline evaluation.

