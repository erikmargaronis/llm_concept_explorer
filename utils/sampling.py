import numpy as np

def normalize_probabilities(probabilities):
    return probabilities / np.sum(probabilities)

def scale_probabilities(probabilities, temperature):
    # Apply temperature
    scaled_probabilities = np.power(probabilities, 1 / temperature)
    scaled_probabilities = scaled_probabilities / np.sum(scaled_probabilities)
    return scaled_probabilities

def top_p_sampling(probs, p):
    """
    Given a 1D numpy array of probabilities,
    retains the minimal number of highest-probability entries
    whose cumulative sum ≥ p, and renormalizes them.

    probs is assumed to be a sorted array of probabilities.
    """

    idx = np.arange(len(probs))

    # Find cutoff
    cumulative = np.cumsum(probs)
    cutoff_idx = np.searchsorted(cumulative, p)

    # Indices to keep
    keep_idx = idx[:cutoff_idx + 1]

    # Create output distribution
    out = np.zeros_like(probs)
    selected_probs = probs[keep_idx]
    out[keep_idx] = selected_probs / selected_probs.sum()

    return out


def sample_next_word(probabilities, candidate_words, k=1):
    sampled_words = np.random.choice(candidate_words, size=k, p=probabilities)
    return sampled_words

def samples_to_probability_distribution(sampled_words, candidate_words):
    probability_distribution = np.zeros(len(candidate_words))
    for word in sampled_words:
        probability_distribution[candidate_words.index(word)] += 1
    return probability_distribution / np.sum(probability_distribution)