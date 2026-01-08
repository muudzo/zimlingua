import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.translator import Translator
from src.utils import logger

st.set_page_config(page_title="ZimLingua", layout="wide")

st.title("🇿🇼 ZimLingua: Neural Translation")
st.markdown("Offline translation for English, Shona, and Ndebele.")

# Initialize backend (cache resource to avoid reloading model)
@st.cache_resource
def load_translator():
    return Translator()

try:
    translator = load_translator()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source")
    src_lang = st.selectbox("Source Language", ["English (en)", "Shona (sn)", "Ndebele (nd)"], index=0)
    source_text = st.text_area("Enter text to translate:", height=200)

with col2:
    st.subheader("Target")
    tgt_lang = st.selectbox("Target Language", ["English (en)", "Shona (sn)", "Ndebele (nd)"], index=1)
    
    translate_btn = st.button("Translate", type="primary")
    
    if translate_btn and source_text:
        # Extract short codes
        src_code = src_lang.split("(")[1].split(")")[0]
        tgt_code = tgt_lang.split("(")[1].split(")")[0]
        
        # Map codes
        src_flores = translator.get_language_code(src_code)
        tgt_flores = translator.get_language_code(tgt_code)
        
        with st.spinner("Translating..."):
            try:
                results = translator.translate_batch([source_text], source_lang=src_flores, target_lang=tgt_flores)
                st.text_area("Translation:", value=results[0], height=200)
            except Exception as e:
                st.error(f"Translation error: {e}")

st.markdown("---")
st.caption("Powered by NLLB-200 & CTranslate2 | Built with Streamlit")
