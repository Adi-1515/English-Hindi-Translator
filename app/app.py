import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import os
import time

# --- Setup & Config ---
st.set_page_config(
    page_title="Eng↔Hin Neural Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
CUSTOM_CSS = """
<style>
/* Import Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Sans+Devanagari:wght@400;500;600&display=swap');
@import url('https://unpkg.com/@phosphor-icons/web@2.1.1/src/regular/style.css');

/* Base Styles */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}

/* General adjustments */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 80rem; /* 7xl */
}

/* Styling elements to look like Tailwind */
.stTextArea textarea {
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    background-color: #ffffff;
    padding: 1rem;
    min-height: 250px;
    font-size: 1rem;
}
.stTextArea textarea:focus {
    border-color: #64748b;
    box-shadow: 0 0 0 2px rgba(100, 116, 139, 0.2);
}

/* Devanagari font class for Hindi */
.font-devanagari textarea, .font-devanagari div {
    font-family: 'Noto Sans Devanagari', sans-serif !important;
}

/* Primary Button Styling (Translate) */
.stButton>button[kind="primary"] {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: none;
    border-radius: 0.375rem;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    font-size: 0.875rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    transition: all 0.2s;
}
.stButton>button[kind="primary"]:hover {
    background-color: #334155 !important;
    color: #ffffff !important;
}

/* Swap Button styling */
.swap-btn > button {
    background-color: transparent !important;
    color: #64748b !important;
    border: 1px solid #e2e8f0;
    padding: 0.5rem;
    width: 100%;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}
.swap-btn > button:hover {
    background-color: #f1f5f9 !important;
}

/* Header styling */
.header-container {
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.header-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.header-subtitle {
    font-size: 1rem;
    color: #64748b;
    font-weight: 500;
}

/* Output Box Styling */
.output-box {
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
    background-color: #f8fafc;
    min-height: 250px;
    padding: 1rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}
.output-placeholder {
    color: #94a3b8;
}

/* Examples buttons */
.example-btn > button {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
    text-align: left;
    justify-content: flex-start;
    padding: 0.75rem;
    height: auto;
    width: 100%;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}
.example-btn > button:hover {
    border-color: #94a3b8 !important;
}

/* History and Model Info */
.info-card {
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
    background-color: #ffffff;
    padding: 1.25rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    height: 100%;
}
.history-item {
    padding: 0.75rem;
    border: 1px solid #f1f5f9;
    border-radius: 0.25rem;
    margin-bottom: 0.5rem;
    background-color: #ffffff;
}

/* Subheaders */
.section-header {
    font-size: 0.875rem;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 1rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model(direction):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    if direction == "en-hi":
        model_dir = os.path.join(base_dir, "model")
        if not os.path.exists(model_dir):
            fallback_dir = os.path.join(base_dir, "tf_model")
            if os.path.exists(fallback_dir):
                model_dir = fallback_dir
            else:
                st.error(f"Model directory not found at {model_dir}. Please train or download the model first.")
                st.stop()
    else: # hi-en
        model_dir = os.path.join(base_dir, "model_hi_en")
        if not os.path.exists(model_dir):
            fallback_dir = os.path.join(base_dir, "tf_model_hi_en")
            if os.path.exists(fallback_dir):
                model_dir = fallback_dir
            else:
                model_dir = "Helsinki-NLP/opus-mt-hi-en"
            
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return tokenizer, model

# --- State Initialization ---
if 'direction' not in st.session_state:
    st.session_state.direction = 'en-hi'
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'source_input' not in st.session_state:
    st.session_state.source_input = ""
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
if 'history' not in st.session_state:
    st.session_state.history = []

def swap_languages():
    st.session_state.direction = 'hi-en' if st.session_state.direction == 'en-hi' else 'en-hi'
    # Swap text
    temp = st.session_state.get('source_input', st.session_state.input_text)
    st.session_state.source_input = st.session_state.translated_text
    st.session_state.input_text = st.session_state.translated_text
    st.session_state.translated_text = temp

def translate():
    text = st.session_state.get('source_input', st.session_state.input_text).strip()
    if not text:
        st.session_state.translated_text = ""
        return
    
    tokenizer, model = load_model(st.session_state.direction)
    tokenized_input = tokenizer([text], return_tensors='tf')
    generated_tokens = model.generate(**tokenized_input, max_length=128)
    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    st.session_state.translated_text = translation
    
    # Add to history
    st.session_state.history.insert(0, {
        "src": text,
        "tgt": translation,
        "dir": st.session_state.direction
    })
    # Keep only last 5
    st.session_state.history = st.session_state.history[:5]

def set_example(src, tgt):
    st.session_state.source_input = src
    st.session_state.input_text = src
    st.session_state.translated_text = tgt

# --- Header ---
st.markdown("""
<div class="header-container">
    <div>
        <div class="header-title"><i class="ph ph-translate" style="color: #475569; font-size: 1.75rem;"></i> Eng↔Hin Translator</div>
        <div class="header-subtitle">Neural Machine Translation</div>
    </div>
    <div style="display:flex; gap: 1rem; align-items:center; font-size: 0.875rem; font-weight: 500; color: #475569;">
        <a href="https://huggingface.co/Helsinki-NLP" target="_blank" style="text-decoration:none; color:inherit; cursor:pointer; transition: color 0.2s;" onmouseover="this.style.color='#0f172a'" onmouseout="this.style.color='inherit'">About</a>
        <a href="https://huggingface.co/Helsinki-NLP/opus-mt-en-hi" target="_blank" style="text-decoration:none; color:inherit; cursor:pointer; transition: color 0.2s;" onmouseover="this.style.color='#0f172a'" onmouseout="this.style.color='inherit'">Model</a>
        <a href="https://github.com/Adi-1515/English-Hindi-Translator.git" target="_blank" style="text-decoration:none; color:inherit; cursor:pointer; transition: color 0.2s;" onmouseover="this.style.color='#0f172a'" onmouseout="this.style.color='inherit'"><i class="ph ph-github-logo" style="font-size: 1.25rem;"></i></a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Workspace ---
src_lang = "English" if st.session_state.direction == 'en-hi' else "Hindi"
tgt_lang = "Hindi" if st.session_state.direction == 'en-hi' else "English"

# Language Controls
col1, col2, col3, col_pad = st.columns([1, 0.5, 1, 8])
with col1:
    st.markdown(f"<div style='text-align:center; padding:0.4rem; font-size:0.875rem; font-weight:500; border:1px solid #e2e8f0; border-radius:0.25rem; background-color: #ffffff; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);'>{src_lang}</div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="swap-btn">', unsafe_allow_html=True)
    st.button("⇄", on_click=swap_languages, help="Swap languages", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown(f"<div style='text-align:center; padding:0.4rem; font-size:0.875rem; font-weight:500; border:1px solid #e2e8f0; border-radius:0.25rem; background-color: #ffffff; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);'>{tgt_lang}</div>", unsafe_allow_html=True)

st.write("") # spacing

# Editor Grid
edit_col1, edit_col2 = st.columns(2)

with edit_col1:
    st.markdown(f"<div style='font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 0.5rem; padding-left: 0.25rem;'>{src_lang}</div>", unsafe_allow_html=True)
    
    text_input = st.text_area(
        label="Source text",
        placeholder=f"Enter text in {src_lang}...",
        label_visibility="collapsed",
        key="source_input",
    )
    if text_input != st.session_state.input_text:
        st.session_state.input_text = text_input

with edit_col2:
    st.markdown(f"<div style='font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 0.5rem; padding-left: 0.25rem;'>{tgt_lang}</div>", unsafe_allow_html=True)
    
    output_text = st.session_state.translated_text
    
    text_class = "font-devanagari" if tgt_lang == "Hindi" else ""
    text_size = "1.125rem" if tgt_lang == "Hindi" else "1rem"
    
    if output_text:
        st.markdown(f"""
        <div class="output-box {text_class}">
            <div style="font-size: {text_size}; color: #0f172a; white-space: pre-wrap;">{output_text}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="output-box">
            <div class="output-placeholder font-sans">Your {tgt_lang} translation will appear here...</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")
col_btn, _ = st.columns([1, 6])
with col_btn:
    st.button("Translate", on_click=translate, type="primary", use_container_width=True)

st.divider()

# --- Examples Section ---
st.markdown('<div class="section-header">Try an example</div>', unsafe_allow_html=True)
ex1, ex2, ex3 = st.columns(3)

# Define example pairs
examples = [
    ("Where is the nearest railway station?", "निकटतम रेलवे स्टेशन कहाँ है?"),
    ("Machine learning is changing the way we solve complex problems.", "मशीन लर्निंग हमारे जटिल समस्याओं को हल करने के तरीके को बदल रही है।"),
    ("Hello, how are you today?", "नमस्ते, आज आप कैसे हैं?")
]

with ex1:
    st.markdown('<div class="example-btn">', unsafe_allow_html=True)
    src_text = examples[0][0] if st.session_state.direction == 'en-hi' else examples[0][1]
    tgt_text = examples[0][1] if st.session_state.direction == 'en-hi' else examples[0][0]
    st.button(src_text, use_container_width=True, key="ex1", on_click=set_example, args=(src_text, tgt_text))
    st.markdown('</div>', unsafe_allow_html=True)
    
with ex2:
    st.markdown('<div class="example-btn">', unsafe_allow_html=True)
    src_text = examples[1][0] if st.session_state.direction == 'en-hi' else examples[1][1]
    tgt_text = examples[1][1] if st.session_state.direction == 'en-hi' else examples[1][0]
    st.button(src_text, use_container_width=True, key="ex2", on_click=set_example, args=(src_text, tgt_text))
    st.markdown('</div>', unsafe_allow_html=True)
    
with ex3:
    st.markdown('<div class="example-btn">', unsafe_allow_html=True)
    src_text = examples[2][0] if st.session_state.direction == 'en-hi' else examples[2][1]
    tgt_text = examples[2][1] if st.session_state.direction == 'en-hi' else examples[2][0]
    st.button(src_text, use_container_width=True, key="ex3", on_click=set_example, args=(src_text, tgt_text))
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# --- Lower Info Area ---
lower1, lower2 = st.columns(2)

with lower1:
    st.markdown('<div class="section-header">Translation History</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("<p style='font-size: 0.875rem; color: #64748b; font-style: italic;'>No recent translations.</p>", unsafe_allow_html=True)
    else:
        for item in st.session_state.history:
            src_family = "sans-serif" if item['dir'] == 'en-hi' else "'Noto Sans Devanagari', sans-serif"
            tgt_family = "'Noto Sans Devanagari', sans-serif" if item['dir'] == 'en-hi' else "sans-serif"
            
            st.markdown(f"""
            <div class="history-item">
                <div style="font-family: {src_family}; font-size: 0.875rem; color: #334155; margin-bottom: 0.25rem;">{item['src']}</div>
                <div style="font-family: {tgt_family}; font-size: 0.875rem; color: #0f172a; font-weight: 500;">{item['tgt']}</div>
            </div>
            """, unsafe_allow_html=True)

with lower2:
    st.markdown("""
    <div class="info-card">
        <div style="font-size: 0.875rem; font-weight: 600; color: #0f172a; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
            <i class="ph ph-cpu" style="font-size: 1.125rem; color: #475569;"></i> About the Model
        </div>
        <p style="font-size: 0.875rem; color: #475569; margin-bottom: 1rem; line-height: 1.5;">
            This interface demonstrates integration with a pretrained neural machine translation (NMT) model specialized for the English-Hindi language pair.
        </p>
        <div style="background-color: #f8fafc; border: 1px solid #f1f5f9; padding: 0.75rem; border-radius: 0.25rem; font-family: monospace; font-size: 0.75rem; color: #334155;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.35rem; margin-bottom: 0.35rem;">
                <span style="color: #64748b;">Direction</span> <span>English ↔ Hindi</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.35rem; margin-bottom: 0.35rem;">
                <span style="color: #64748b;">Architecture</span> <span>MarianMT (Transformer)</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #64748b;">Framework</span> <span>Hugging Face</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.write("")
st.markdown("""
<div style="border-top: 1px solid #e2e8f0; padding-top: 1.5rem; margin-top: 2rem; display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b;">
    <div style="display:flex; align-items:center; gap: 0.5rem;">
        <i class="ph ph-translate" style="font-size: 1rem;"></i> Eng↔Hin Neural Translator Interface. Built for functional demonstration.
    </div>
    <div style="display: flex; gap: 1rem;">
        <a href="#" style="text-decoration:none; color:inherit; cursor:pointer; transition: color 0.2s;" onmouseover="this.style.color='#0f172a'" onmouseout="this.style.color='inherit'">Privacy</a>
        <a href="#" style="text-decoration:none; color:inherit; cursor:pointer; transition: color 0.2s;" onmouseover="this.style.color='#0f172a'" onmouseout="this.style.color='inherit'">Terms</a>
        <a href="#" style="text-decoration:none; color:inherit; cursor:pointer; transition: color 0.2s;" onmouseover="this.style.color='#0f172a'" onmouseout="this.style.color='inherit'">API Specs</a>
    </div>
</div>
""", unsafe_allow_html=True)
