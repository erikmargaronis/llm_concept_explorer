import numpy as np

def normalize_probabilities(probabilities):
    return probabilities / np.sum(probabilities)

def scale_probabilities(probabilities, temperature):
    # Apply temperature
    scaled_probabilities = np.power(probabilities, 1 / temperature)
    scaled_probabilities = scaled_probabilities / np.sum(scaled_probabilities)
    return scaled_probabilities

def top_k_scaling(probs, k):
    """
    Given a 1D numpy array of probabilities,
    retains the k most probable tokens and sets the rest to zero.
    """
    idx = np.arange(len(probs))
    keep_idx = idx[:k]
    out = np.zeros_like(probs)
    out[keep_idx] = probs[keep_idx]
    out = out / np.sum(out)
    return out

def top_p_scaling(probs, p):
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

def min_p_scaling(probs: np.ndarray, p: float):
    """
    Takes a 1D numpy array of probabilities and applies minimum-p scaling.
    Returns a new numpy array of normalized probabilities with the same length,
    zeroing out entries below the threshold.
    """
    max_prob = probs.max()
    threshold = max_prob * p

    # Mask out probabilities below the threshold
    mask = probs >= threshold
    filtered = probs * mask

    total = filtered.sum()
    if total == 0:
        # If everything got filtered out, return a uniform zero vector
        return np.zeros_like(probs)

    return filtered / total


def sample_next_word(probabilities, candidate_words, k=1):
    sampled_words = np.random.choice(candidate_words, size=k, p=probabilities)
    return sampled_words

def samples_to_probability_distribution(sampled_words, candidate_words):
    probability_distribution = np.zeros(len(candidate_words))
    for word in sampled_words:
        probability_distribution[candidate_words.index(word)] += 1
    return probability_distribution / np.sum(probability_distribution)
