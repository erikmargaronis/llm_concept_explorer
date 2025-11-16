import streamlit as st
import numpy as np

from utils.plot import (
    plot_bar_chart_probability_distribution,
    plot_distribution
)
from utils.sampling import (
    scale_probabilities,
    sample_next_word,
    samples_to_probability_distribution,
    min_p_scaling,
    top_k_scaling,
    top_p_scaling
)

text_prefix = "I have a"
#vocabulary = ["cat", "dog", "house", "car", "dream", "pen", "book", "friend", "idea", "problem"]
#probabilities = np.array([0.2, 0.18, 0.15, 0.12, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04])

vocabulary = ["dog", "cat", "dream", "book", "question", "plan", "car", "pen", "pet", "idea"]
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
st.markdown("""In the plot below, we can see the initial probability distribution over the vocabulary for the next token, given our prompt.""")
plot_distribution(
    probabilities,
    vocabulary,
    #title = "Initial Probability Distribution",
)
st.divider()
st.subheader("Random Sampling")
st.markdown("""In modern LLM's, the defacto standard method to select the next token is to randomly sample the probability distribution over the model's vocabulary.

Random sampling simply draws from the categorical distribution defined by the model's probabilities at each timestep. Each token's chance of being picked is precisely its probability.

Other methods beside random sampling exist, with **Beam Search** and **Greedy Search** being the most prominent. These methods have been, and still are, often used in text generation. However they are not supported by the major LLM providers API's. OpenAI, Google, Anthropic and Ollama all use random sampling.""")
st.markdown("""It's common to adjust the model’s probabilities before sampling because the raw distribution often contains modes that are either too dominant or too noisy for practical generation. Extremely peaked distributions can collapse into repetitive or deterministic outputs, while overly flat ones can amplify rare or incoherent tokens. By reshaping the distribution we can better control the balance between diversity and reliability. Let's explore some common ways to handle this.""")

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
    vocabulary,
    title = "Initial vs. Temperature-Scaled Probability Distribution"
)

st.divider()
st.subheader("Top-k Sampling")
st.markdown(r"""**Top-k sampling** chooses from only the **k most probable tokens** (then re-normalizes). This method sets probabilities of all other tokens to zero and redistributes among the top k.

For instance, if \( k = 3 \), we only consider the top 3 tokens by probability and ignore the rest.""")

top_k_val = st.slider("Top-k", min_value=1, max_value=10, value=10)
modified_probabilities = top_k_scaling(probabilities, top_k_val)

plot_bar_chart_probability_distribution(
    probabilities,
    modified_probabilities,
    vocabulary,
    title = "Initial vs. Top-k Sampling"
)

st.divider()
st.subheader("Top-p (Nucleus) Sampling")
st.markdown(r"""**Top-p (or nucleus) sampling** chooses from the **smallest set of tokens whose cumulative probability exceeds p**. In other words:
1. Sort tokens by probability descending.
2. Keep adding tokens to the set until the sum of probabilities exceeds \( p \).
3. Zero out everything else, and re-normalize.

This method adapts dynamically to how peaked the distribution is. For a very peaked distribution, the top tokens might already exceed \(p\). For a flatter distribution, we might need more tokens.""")

top_p_val = st.slider("Top-p", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
modified_probabilities = top_p_scaling(probabilities, top_p_val)

plot_bar_chart_probability_distribution(
    probabilities,
    modified_probabilities,
    vocabulary,
    title = "Initial vs. Top-p Sampling"
)

st.divider()
st.subheader("Min-p Sampling")
st.markdown(r"""[Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](https://arxiv.org/abs/2407.01082)

The core idea of min-p sampling is to dynamically adjust the sampling threshold based on the model's confidence at each decoding step.

- Find the highest probability at a given timestep $p_{\text{max}}$
- Define a relative probability threshold $p_{\text{base}}$
- Calculate the current threshold $p_{\text{scaled}} = p_{\text{max}} \times p_{\text{base}}$
- Discard tokens with a probability lower than $p_{\text{scaled}}$

**Example:** the highest probability at a given timestep $p_{\text{max}}$ is .4 and the relative probability threshold $p_{\text{base}}$ is .05.\
We obtain the current threshold by multiplying them $.4 \times .05 = .02$.""",
unsafe_allow_html=True
)

min_p_val = st.slider("Min-p", min_value=.0, max_value=1.0, value=.0, step=0.01)

modified_probabilities = min_p_scaling(probabilities, min_p_val)

plot_bar_chart_probability_distribution(
    probabilities,
    modified_probabilities,
    vocabulary,
    title = "Initial vs. Min-p Sampling"
)

st.divider()
st.subheader("Summary")
st.markdown("""In text generation, models usually choose the next token according to its probability. Adjusting how that choice is made can shift the output toward tighter structure or more open-ended variation. Several sampling strategies exist, each shaping the balance between stability and inventiveness in different ways.""")
st.markdown(r"""| Method | Summary | Strength | Weakness |
| ------ | ------ | ------ | ------ |
| **Temperature** | Sharpens or flattens the global distribution. | Provides flexible control over randomness, enabling both deterministic and creative outputs. | Global adjustments can lead to oversampling unlikely tokens or undersampling diverse options in nuanced contexts. |
| **Top-k** | Keep a fixed number of most probable tokens, discard the rest. | Prevents sampling from long-tail, low-probability tokens, ensuring coherent and focused output. | Keeps a fixed number of tokens, even if some in the top $k$ have extremely low probabilities, which can lead to suboptimal choices in contexts with skewed distributions. |
| **Top-p** | Dynamically adapt the sampling pool to include only the most probable tokens. | Adapts the token pool to context, ensuring more flexibility and diversity while maintaining coherence. | Can lead to overly small or overly large sampling pools depending on the distribution, making it less predictable than fixed-size methods like top-k. |
| **Min-p** | Dynamically adapt the sampling pool to include only relatively probable tokens wrt the most probable one. | Balances coherence and creativity by making token thresholds context-sensitive, avoiding arbitrary truncation and enabling diverse generation in uncertain contexts. | Requires careful tuning of the $p_{\text{base}}$ parameter, and its reliance on relative probabilities may sometimes overlook rare but contextually valid tokens. |""")
