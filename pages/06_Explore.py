import numpy as np
import streamlit as st

def load_embeddings():
    embeddings = np.load("assets/token_embeddings.npy")
    with open("assets/tokens.txt", "r") as f:
        tokens = f.read().splitlines()
    return embeddings, tokens

embeddings, tokens = load_embeddings()
token_ids = [
    976,
    4853,
    19705,
    68347,
    65613,
    1072,
    290,
    29082,
    6446,
    13
]

embeddings_as_str = []
for e in embeddings:
    embeddings_as_str.append('<br>'.join([f'{v:.3f}' for v in e.tolist()]))

pill_css = """
<style>
.input-text {
    font-size: 1.5rem;
    grid-column: 1 / -1;
    text-align: center;
    line-height: 1;
}

.pill-layout {
    display: grid;
    column-gap: 0.5rem;
    row-gap: 0.5rem;
    width: 100%;
}

.pill-cell {
    display: flex;
    justify-content: center;
    align-items: center;
}

.pill {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 10px;
    background-color: #f0f2f6;
    border: 1px solid #d0d3d8;
    font-size: 0.9rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}

.embeddings {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 10px;
    background-color: #f0f2f6;
    border: 1px solid #d0d3d8;
    font-size: 0.6rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}

.arrow-row {
    grid-column: 1 / -1;
    text-align: center;
    font-size: 36px;
    line-height: 1;
}

.fixed-button-container {
    width: 100px;       /* set desired width */
    margin: 0 auto;     /* center horizontally */
}
</style>
"""
st.markdown(pill_css, unsafe_allow_html=True)

# --- STEP STATE SETUP ---
if "step" not in st.session_state:
    # 0 = only input
    # 1 = show tokens
    # 2 = show token ids
    # 3 = show embeddings
    st.session_state.step = 0

max_step = 3

def build_html(step: int) -> str:
    """Build the HTML up to the given step."""
    n = len(tokens)
    html = f'<div class="pill-layout" style="grid-template-columns: repeat({n}, 1fr);">'

    # Step 0: input text only
    html += '<div class="input-text">The quick brown fox jumps over the lazy dog.</div>'

    # Step 1: tokens
    if step >= 1:
        html += '<div class="arrow-row">&#x2193;</div>'
        for t in tokens:
            html += f'<div class="pill-cell"><div class="pill">{t}</div></div>'

    # Step 2: token ids
    if step >= 2:
        html += '<div class="arrow-row">&#x2193;</div>'
        for tid in token_ids:
            html += f'<div class="pill-cell"><div class="pill">{tid}</div></div>'

    # Step 3: embeddings
    if step >= 3:
        html += '<div class="arrow-row">&#x2193;</div>'
        for e in embeddings_as_str:
            html += f'<div class="pill-cell"><div class="embeddings">{e}</div></div>'

    html += "</div>"
    return html


left, middle, right = st.columns([1, 1, 1])

with middle:
    step_label = ["Tokenize", "Lookup IDs", "Lookup Embeddings", "Reset"]

    with st.container():
        # Inject CSS scoped to this column block
        st.markdown("""
        <style>
        /* Target the first stButton inside this block only */
        div[data-testid="stVerticalBlock"] div.stButton > button {
            width: 170px;
            height: 40px;
        }
        </style>
        """, unsafe_allow_html=True)

        clicked = st.button(step_label[st.session_state.step])
        st.markdown('</div>', unsafe_allow_html=True)

        if clicked:
            if st.session_state.step == 3:
                st.session_state.step = 0
                st.rerun()
            elif st.session_state.step < max_step:
                st.session_state.step += 1
                st.rerun()

placeholder = st.empty()

html = build_html(st.session_state.step)
placeholder.markdown(html, unsafe_allow_html=True)
