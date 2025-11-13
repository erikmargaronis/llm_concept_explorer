import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
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

# Ensure probabilities sum to 1 (due to potential floating point inaccuracies)
probabilities = probabilities / np.sum(probabilities)

temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

def perform_sampling(probabilities, temperature):
    # Apply temperature
    scaled_probabilities = np.power(probabilities, 1 / temperature)
    scaled_probabilities = scaled_probabilities / np.sum(scaled_probabilities)
    return scaled_probabilities

# Call perform_sampling outside the button click to update dynamically
modified_probabilities = perform_sampling(probabilities, temperature)

# Update the plot dynamically
fig_combined, ax_combined = plt.subplots(figsize=(10, 6))

# Plot initial probabilities as dotted bars
ax_combined.bar(candidate_words, probabilities, color='none', label='Initial Distribution', edgecolor='black', linestyle='--')

if modified_probabilities is not None:
    # Plot modified probabilities as regular bars
    ax_combined.bar(candidate_words, modified_probabilities, color='skyblue', edgecolor='black', label='Temperature-Scaled Distribution', alpha=0.7)

ax_combined.set_xlabel("Candidate Words")
ax_combined.set_ylabel("Probability")
ax_combined.set_title("Initial vs. Temperature-Scaled Probability Distribution")
ax_combined.tick_params(axis='x', rotation=45)
ax_combined.legend(loc='upper right')
plt.tight_layout()
ax_combined.set_ylim(0, 1) # Set y-axis limits from 0 to 1
st.pyplot(fig_combined)
plt.close(fig_combined) # Close the plot to free up memory

if st.button("Sample Next Word"):    
    modified_probabilities_on_click = perform_sampling(probabilities, temperature)

    if modified_probabilities_on_click is not None and np.sum(modified_probabilities_on_click) > 0:
        # Filter out words with zero probability before sampling
        non_zero_prob_indices = np.where(modified_probabilities_on_click > 0)[0]
        if non_zero_prob_indices.size > 0:
            valid_words_for_sampling = [candidate_words[i] for i in non_zero_prob_indices]
            valid_probabilities_for_sampling = modified_probabilities_on_click[non_zero_prob_indices]
            sampled_word = np.random.choice(valid_words_for_sampling, p=valid_probabilities_for_sampling / np.sum(valid_probabilities_for_sampling))
            st.write(f"**{sampled_word}**")
        else:
            st.write("No word could be sampled with non-zero probability.")
    else:
        st.write("No word could be sampled with the given parameters on button click.")


