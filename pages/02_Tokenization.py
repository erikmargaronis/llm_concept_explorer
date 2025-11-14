import streamlit as st
import tiktoken
import html  # NEW

#st.set_page_config(layout="wide")
enc = tiktoken.encoding_for_model("gpt-4o")

st.title("Tokenization")
st.markdown("""Tokenization is the process of breaking text into small pieces—called tokens—that a language model can understand and work with.A token might be a whole word, part of a word, or even punctuation. Models don’t read text the way people do; they work with numbers. Tokenization provides a predictable way to turn text into numerical units the model can process.

This step is needed because it keeps the input consistent, manageable, and efficient. By using tokens instead of full sentences or paragraphs, the model can handle many languages, deal with rare or invented words, and operate at high speed. Tokenization is the bridge between human language and the numerical world in which these models operate.""")

left, right = st.columns([1.1, 1])

with left:
    text_input = st.text_area(
        "Text",
        height=300,
        placeholder="Type or paste text..."
    )

    if text_input:
        tokens = enc.encode(text_input)
        st.metric("Token count", len(tokens))

with right:
    if text_input:

        def token_color(t):
            colors = [
                "#ff6666", "#ff9966", "#ffcc66",
                "#99cc66", "#66cccc", "#6699ff",
                "#cc66ff"
            ]
            return colors[t % len(colors)]

        colored_tokens = []
        for t in tokens:
            piece = enc.decode([t]).replace("\n", "\\n")
            # Escape HTML and preserve spaces
            piece = html.escape(piece).replace(" ", "&nbsp;")
            colored_tokens.append(
                f"<span title='{t}' style='background:{token_color(t)}22; "
                f"padding:2px 4px; border-radius:4px; "
                f"display:inline-block; margin:2px;'>"
                f"{piece}"
                "</span>"
            )


        st.markdown(
            f"""
            <div style="
                border:1px solid #ddd; 
                padding:10px; 
                border-radius:6px;
                height:350px; 
                overflow-y:scroll;
                background:#fafafa;
                white-space: pre-wrap;
                ">
                {''.join(colored_tokens)}
            </div>
            """,
            unsafe_allow_html=True
        )

        token_pairs = [
            f"`{enc.decode([t])}` → **{t}**"
            for t in tokens
        ]
        st.markdown("<br>".join(token_pairs), unsafe_allow_html=True)
