import streamlit as st
import numpy as np

from utils.sampling import scale_probabilities, sample_next_word, samples_to_probability_distribution, top_p_sampling
from utils.plot import plot_bar_chart_probability_distribution, plot_distribution

text_prefix = "I have a"
#candidate_words = ["cat", "dog", "house", "car", "dream", "pen", "book", "friend", "idea", "problem"]
#probabilities = np.array([0.2, 0.18, 0.15, 0.12, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04])

candidate_words = ["dog", "cat", "dream", "book", "question", "plan", "car", "pen", "pet", "idea"]
probabilities = np.array([0.45853809710312615, 0.40465845041054327, 0.07031908764647739, 0.015690313995143465, 0.015690313995143465, 0.012219628826053952, 0.010783784589726049, 0.0050938991521505715, 0.002406187582511743, 0.0006893842845350389])

probabilities = probabilities / np.sum(probabilities)


st.title("Understanding Sampling in LLMs")
st.markdown("""Large Language Models (LLMs) produce text by predicting the next token (word, sub-word, or character) given a context. Under the hood, they produce a probability distribution over a vocabulary of possible next tokens. But how do we go from probabilities to an actual token choice? That’s where **sampling** comes in.""")

st.subheader("Vocabulary and Probability Distributions")
st.markdown("""OpenAI's Chat Completions API provides a parameter, `top_logprobs`, which allows us to retrieve the **n most likely tokens** for the next timestep, along with their associated log probabilities.

- We'll use this parameter to extract the ten most probable tokens and their log probabilities for the next timestep, given a specific prompt.
- We'll treat these ten tokens as our vocabulary to explore and illustrate different sampling methods in a simplified context.
- Note that in practice the whole model vocabulary is used, which is much larger than our ten samples. Typically in the range of 60.000-100.000+ tokens.""")

st.markdown("""**Example prompt:** **`I have a`**""")
st.markdown("""**Possible next words:** **`dog`, `cat`, `dream`, `book`, `question`, `plan`, `car`, `pen`, `pet`, `idea`**""")

plot_distribution(
    probabilities,
    "Initial Probability Distribution",
    candidate_words
)
st.divider()
st.subheader("Random Sampling")
st.markdown("""In modern LLM's, the defacto standard method to select the next token is to randomly sample the probability distribution over the model's vocabulary.

Random sampling simply draws from the categorical distribution defined by the model's probabilities at each timestep. Each token's chance of being picked is precisely its probability.

Other methods beside random sampling exist, with **Beam Search** and **Greedy Search** being the most prominent. These methods have been, and still are, often used in text generation. However they are not supported by the major LLM providers API's. OpenAI, Google, Anthropic and Ollama all use random sampling.""")

st.divider()
st.subheader("Temperature-Scaled Sampling")

st.markdown(
r"""Temperature is a way to sharpen or flatten the distribution before sampling. This is done by dividing the *logits* (raw scores) before the softmax function:


$$
\hat{y}_i = \text{softmax}\left(\frac{z_i}{T}\right) = \frac{\exp\left(\frac{z_i}{T}\right)}{\sum_{j} \exp\left(\frac{z_j}{T}\right)}
$$

- If $T<1$, high-probability tokens get amplified (distribution gets sharper).
- If $T>1$, probabilities get "flattened," increasing randomness.

This means that with a lower temperature, the generations are more likely to be coherent and predictable. Conversely, at higher temperatures the generations will appear more random, or *creative*.""",
unsafe_allow_html=True
)

temperature = st.slider("Temperature", min_value=0.01, max_value=2.0, value=1.0, step=0.01)

modified_probabilities = scale_probabilities(probabilities, temperature)

plot_bar_chart_probability_distribution(
    probabilities,
    modified_probabilities,
    candidate_words,
    title = "Initial vs. Temperature-Scaled Probability Distribution"
    )

if False:
    num_samples = st.slider("Number of Samples", min_value=1, max_value=100, value=1)

    if st.button("Sample Next Word(s)", key="sample_next_word"):
        sampled_words = sample_next_word(modified_probabilities, candidate_words, k=num_samples)
        probability_distribution = samples_to_probability_distribution(sampled_words, candidate_words)
        print(sampled_words)
        print(probability_distribution)

        plot_bar_chart_probability_distribution(probability_distribution, modified_probabilities, candidate_words)

st.divider()
st.subheader("Top-p (Nucleus) Sampling")

st.markdown(r"""**Top-p (or nucleus) sampling** chooses from the **smallest set of tokens whose cumulative probability exceeds p**. In other words:
1. Sort tokens by probability descending.
2. Keep adding tokens to the set until the sum of probabilities exceeds \( p \).
3. Zero out everything else, and re-normalize.

This method adapts dynamically to how peaked the distribution is. For a very peaked distribution, the top tokens might already exceed \(p\). For a flatter distribution, we might need more tokens.""")

top_p_val = st.slider("Top-p", min_value=0.01, max_value=1.0, value=1.0, step=0.01)

modified_probabilities = top_p_sampling(probabilities, top_p_val)

plot_bar_chart_probability_distribution(
    probabilities,
    modified_probabilities,
    candidate_words,
    title = "Initial vs. Top-p Sampling"
)

st.divider()
st.subheader("Top-k Sampling")