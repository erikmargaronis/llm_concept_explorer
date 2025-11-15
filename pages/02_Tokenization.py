import streamlit as st
import tiktoken
import html

enc = tiktoken.encoding_for_model("gpt-4o")

st.title("Tokenization")
st.markdown("""Tokenization divides text into smaller units `tokens`that a model can interpret. A token can be a full word, a fragment, or a piece of punctuation. Instead of reading text in a human sense, the model processes numerical representations, and tokenization defines how each fragment becomes a number.

This procedure creates a consistent and efficient format for input. Working with tokens allows the system to handle many languages, absorb unusual or invented terms, and maintain speed. It forms the link between written language and the numerical space in which these systems function.""")

st.markdown("""Type some text in the text field below to see how it gets tokenized.""")

left, right = st.columns([1.1, 1])

with left:
    text_input = st.text_area(
        "Text to tokenize",
        height=300,
        placeholder="Type or paste text to tokenize...",
        label_visibility="collapsed"
    )


    if st.button("Tokenize") or text_input:
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

        def format_token_text(t: int) -> str:
            s = enc.decode([t])
            s = s.replace("\n", "\\n")
            s = s.replace("\r", "\\r")
            s = s.replace("`", "\\`")
            return s

        colored_tokens = []
        for t in tokens:
            piece = format_token_text(t)
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
                height:300px; 
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
            f"`{format_token_text(t)}` → **{t}**"
            for t in tokens
        ]
        st.markdown("<br>".join(token_pairs), unsafe_allow_html=True)
