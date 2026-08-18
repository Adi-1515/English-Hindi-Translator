# English to Hindi Translator

A web app that translates English text to Hindi using a pre-trained MarianMT model, fine-tuned on an English-Hindi dataset.

## How it works

1. The training script (`training/train.py`) fine-tunes the `Helsinki-NLP/opus-mt-en-hi` model on 50,000 English-Hindi sentence pairs and saves the weights locally. (You can also use `training/fastTrain.py` for a faster training process on a smaller dataset).
2. The app (`app/app.py`) loads those saved weights and runs a Streamlit web interface where you can type English text and get a Hindi translation.

## Tech Stack

| Part | Library |
|---|---|
| Model | HuggingFace Transformers (MarianMT) |
| Training | TensorFlow / Keras |
| App / Inference | PyTorch |
| Web interface | Streamlit |
| Dataset | HuggingFace Datasets |

> **Note:** Training uses TensorFlow locally. The deployed app uses PyTorch, since TensorFlow doesn't support Python 3.14 (which Streamlit Cloud uses).

## Project Structure

```
PreTrainedTranslator/
├── app/
│   └── app.py          # Streamlit web interface
├── training/
│   ├── train.py        # Fine-tuning script
│   └── fastTrain.py    # Faster training script (smaller dataset)
├── requirements.txt
└── README.md
```

> Model weights (`model/`) are generated locally after training and are not included in this repo.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Adi-1515/English-Hindi-Translator.git
   cd PreTrainedTranslator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Step 1 — Train the model** *(skip if you already have saved weights)*
```bash
python training/train.py
```
*(Tip: Use `python training/fastTrain.py` instead for a much faster training run using a smaller dataset.)*

This downloads the dataset, fine-tunes the model for 1 epoch, and saves the weights to a `model/` folder.

**Step 2 — Run the app**
```bash
streamlit run app/app.py
```

## Dataset

- **Name:** `cfilt/iitb-english-hindi`
- **Source:** HuggingFace Datasets
- **Details:** A parallel English-Hindi corpus from IIT Bombay. Training uses 50,000 sentences; validation uses 2,000.

## License

See the [LICENSE](LICENSE) file for details.
