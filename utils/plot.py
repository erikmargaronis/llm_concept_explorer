import matplotlib.pyplot as plt
import streamlit as st

def plot_bar_chart_probability_distribution(
    probabilities,
    modified_probabilities,
    candidate_words
):
    fig_combined, ax_combined = plt.subplots(figsize=(8, 4))
    ax_combined.bar(
        candidate_words,
        probabilities,
        color='none',
        label='Initial Distribution',
        edgecolor='black',
        linestyle='--'
    )

    if modified_probabilities is not None:
        ax_combined.bar(
            candidate_words,
            modified_probabilities,
            color='skyblue',
            edgecolor='black',
            label='Temperature-Scaled Distribution',
            alpha=0.7
        )

    ax_combined.set_xlabel("Candidate Words")
    ax_combined.set_ylabel("Probability")
    ax_combined.set_title("Initial vs. Temperature-Scaled Probability Distribution")
    ax_combined.tick_params(axis='x', rotation=45)
    ax_combined.legend(loc='upper right')
    #plt.tight_layout()
    ax_combined.set_ylim(0, 1) # Set y-axis limits from 0 to 1
    #col1, col2 = st.columns(2)
    #with col1:
    st.pyplot(fig_combined)
    plt.close(fig_combined) # Close the plot to free up memory