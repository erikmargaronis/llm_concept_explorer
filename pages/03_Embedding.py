import numpy as np
import streamlit as st
import plotly.express as px
from sympy.polys.rootoftools import I

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
    grid-column: 1 / -1;      /* span all columns */
    text-align: center;
    #font-size: 36px;
    line-height: 1;
}

.pill-layout {
    display: grid;
    column-gap: 0.5rem;
    row-gap: 0.5rem;
    width: 100%;              /* anchored left/right inside Streamlit column */
}

.pill-cell {
    display: flex;
    justify-content: center;  /* center pill horizontally within cell */
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
    max-width: 100%;          /* never wider than its grid cell */
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
    max-width: 100%;          /* never wider than its grid cell */
}

.arrow-row {
    grid-column: 1 / -1;      /* span all columns */
    text-align: center;
    font-size: 36px;
    line-height: 1;
}
</style>
"""
st.markdown(pill_css, unsafe_allow_html=True)

n = len(tokens)  # number of columns

html = f'<div class="pill-layout" style="grid-template-columns: repeat({n}, 1fr);">'

html += '<div class="input-text">The quick brown fox jumps over the lazy dog.</div>'

html += '<div class="arrow-row">&#x2193;</div>'

# Row 1: tokens
for t in tokens:
    html += f'<div class="pill-cell"><div class="pill">{t}</div></div>'

# Row 2: arrow spanning all columns
html += '<div class="arrow-row">&#x2193;</div>'

# Row 3: indices
for tid in token_ids:
    html += f'<div class="pill-cell"><div class="pill">{tid}</div></div>'

html += '<div class="arrow-row">&#x2193;</div>'

for e in embeddings_as_str:
    html += f'<div class="pill-cell"><div class="embeddings">{e}</div></div>'
    #html += f'<p style="font-size:8px;">{e}</p>'

html += "</div>"

st.markdown(html, unsafe_allow_html=True)
