import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import numpy as np
import os

# Configuration
max_length = 128

# Load the trained model and tokenizer
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, "model")
    
    if not os.path.exists(model_dir):
        # Fallback to tf_model if model doesn't exist
        fallback_dir = os.path.join(base_dir, "tf_model")
        if os.path.exists(fallback_dir):
            model_dir = fallback_dir
        else:
            st.error(f"Model directory not found at {model_dir}. Please train or download the model first.")
            st.stop()
            
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit page setup
st.set_page_config(page_title="English to Hindi Translator", page_icon="🌐")
st.title("🌐 English to Hindi Translator")
st.markdown("### Translate English sentences to Hindi using your trained Deep Learning model!")

# User input
input_text = st.text_area("Enter English text:", "", height=150)

# Translate button
if st.button("Translate"):
    if input_text.strip():
        # Tokenize input
        tokenized_input = tokenizer([input_text], return_tensors='np')

        # Generate translation
        generated_tokens = model.generate(**tokenized_input, max_length=max_length)

        # Decode output
        hindi_translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        st.subheader("Translated Hindi Text:")
        st.success(hindi_translation)
    else:
        st.warning("⚠️ Please enter some text to translate.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using TensorFlow, HuggingFace Transformers, and Streamlit.")
