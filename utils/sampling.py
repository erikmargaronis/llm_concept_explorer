import numpy as np

def normalize_probabilities(probabilities):
    return probabilities / np.sum(probabilities)

def scale_probabilities(probabilities, temperature):
    # Apply temperature
    scaled_probabilities = np.power(probabilities, 1 / temperature)
    scaled_probabilities = scaled_probabilities / np.sum(scaled_probabilities)
    return scaled_probabilities


def sample_next_word(probabilities, candidate_words, k=1):
    sampled_words = np.random.choice(candidate_words, size=k, p=probabilities)
    return sampled_words

def samples_to_probability_distribution(sampled_words, candidate_words):
    probability_distribution = np.zeros(len(candidate_words))
    for word in sampled_words:
        probability_distribution[candidate_words.index(word)] += 1
    return probability_distribution / np.sum(probability_distribution)