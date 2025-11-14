import streamlit as st
import numpy as np

from utils.sampling import scale_probabilities, sample_next_word, samples_to_probability_distribution, normalize_probabilities
from utils.plot import plot_bar_chart_probability_distribution

st.title("Sampling Strategies")
st.subheader("Random Sampling")
st.write("""In modern LLM's, the defacto standard method to select the next token is to randomly sample the probability distribution over the model's vocabulary.

Random sampling simply draws from the categorical distribution defined by the model's probabilities at each timestep. Each token's chance of being picked is precisely its probability.

Other methods beside random sampling exist, with Beam Search and Greedy Search being the most prominent. These methods have been, and still are, often used in text generation. However they are not supported by the major LLM providers API's. OpenAI, Google, Anthropic and Ollama all use random sampling.""")

st.divider()

st.write("In the examples on this page, we will use a prefix (prompt): **I have a**")
st.write("""We also use 10 candidate words that represent the most probable next word for the prefix.
In reality, all tokens in the model's vocabulary are candidates for the next word, but for the sake of simplicity, we will only use 10 candidate words.""")

text_prefix = "I have a"
candidate_words = ["cat", "dog", "house", "car", "dream", "pen", "book", "friend", "idea", "problem"]
probabilities = np.array([0.2, 0.18, 0.15, 0.12, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04])

st.subheader("Temperature-Scaled Sampling")

probabilities = probabilities / np.sum(probabilities)

temperature = st.slider("Temperature", min_value=0.01, max_value=2.0, value=1.0, step=0.01)

modified_probabilities = scale_probabilities(probabilities, temperature)


plot_bar_chart_probability_distribution(probabilities, modified_probabilities, candidate_words)

if False:
    num_samples = st.slider("Number of Samples", min_value=1, max_value=100, value=1)

    if st.button("Sample Next Word(s)", key="sample_next_word"):
        sampled_words = sample_next_word(modified_probabilities, candidate_words, k=num_samples)
        probability_distribution = samples_to_probability_distribution(sampled_words, candidate_words)
        print(sampled_words)
        print(probability_distribution)

        plot_bar_chart_probability_distribution(probability_distribution, modified_probabilities, candidate_words)

st.subheader("Top-P Sampling")

st.write("""Top-P sampling is a method of selecting the next word based on the cumulative probability of the most probable words.""")

