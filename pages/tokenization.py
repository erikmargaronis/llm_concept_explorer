import streamlit as st

st.set_page_config(layout="wide")
st.title("Tokenization")
st.write("This section will handle tokenization features.")

text_input = st.text_area("Enter text here:", height=200)

if st.button("Tokenize"):    
    # This is a placeholder for actual tokenization logic
    tokens = text_input.split()
    st.write("### Tokenized Output:")
    st.write(tokens)
