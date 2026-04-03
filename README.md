# English to Hindi Translator

## Overview

A deep learning based Natural Language Processing application for translating English text to Hindi. This project utilizes a sequence-to-sequence transformer model, fine-tuned on an English-Hindi dataset, and features a web interface for real-time translation.

## Features

- High-quality English to Hindi translation using state-of-the-art transformer architecture.
- Full machine learning pipeline, including data preprocessing, tokenization, training, and inference.
- Interactive and lightweight web interface for demonstration and testing.

## Tech Stack

- **Model architecture:** Transformer (MarianMT)
- **Framework:** TensorFlow, Keras
- **NLP Library:** HuggingFace Transformers, HuggingFace Datasets
- **Frontend / Deployment:** Streamlit
- **Data handling:** NumPy

## Project Structure

```text
PreTrainedTranslator/
├── app/
│   └── app.py
├── training/
│   └── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

*(Note: Model weights and directories like `model/` or `tf_model/` are generated locally and excluded from version control.)*

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd PreTrainedTranslator
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train Model:**
   To train the translation model and save weights locally:
   ```bash
   python training/train.py
   ```

2. **Run Application:**
   Once the model has been trained and saved, launch the web interface:
   ```bash
   streamlit run app/app.py
   ```

## Model Details

- **Base Model:** `Helsinki-NLP/opus-mt-en-hi`
- **Architecture:** Marian Machine Translation (MarianMT) framework, which utilizes sequence-to-sequence transformer architecture optimized for neural machine translation.

## Dataset

- **Name:** `cfilt/iitb-english-hindi`
- **Source:** HuggingFace Datasets
- **Details:** Compiled by the Center for Indian Language Technology (CFILT) at IIT Bombay. It is a comprehensive parallel corpus designed for English-Hindi translation tasks.

## Placeholders

### Screenshots
<!-- 
*(Replace this section with screenshots of your Streamlit web interface)*

`[Screenshot 1 Placeholder: Main Application Screen]`

`[Screenshot 2 Placeholder: Example Translation Output]` -->

## Future Improvements

- Add support for bidirectional translation (Hindi to English).
- Implement advanced batch processing on the inference pipeline.
- Include beam search decoding parameters in the frontend interface.
- Deploy the model to a scalable cloud architecture.

## License

See the [LICENSE](LICENSE) file for details.
