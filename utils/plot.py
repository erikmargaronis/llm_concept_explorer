import matplotlib.pyplot as plt
import streamlit as st

def plot_distribution(probs, vocab, title = ""):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(probs)), probs, tick_label=vocab, alpha=0.7, color="skyblue", edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Token")
    ax.set_ylabel("Probability")

    st.pyplot(fig)
    plt.close(fig)

def plot_bar_chart_probability_distribution(
    probabilities,
    modified_probabilities,
    vocab,
    title = "",
    xlabel = "Token",
    ylabel = "Probability"
):
    fig_combined, ax_combined = plt.subplots(figsize=(8, 4))
    ax_combined.bar(
        vocab,
        probabilities,
        color='none',
        label='Initial Distribution',
        edgecolor='black',
        linestyle='--'
    )

    if modified_probabilities is not None:
        ax_combined.bar(
            vocab,
            modified_probabilities,
            color='skyblue',
            edgecolor='black',
            label='Modified Distribution',
            alpha=0.7
        )

    ax_combined.set_xlabel(xlabel)
    ax_combined.set_ylabel(ylabel)
    ax_combined.set_title(title)
    ax_combined.tick_params(axis='x', rotation=45)
    ax_combined.legend(loc='upper right')
    ax_combined.set_ylim(0, 1) # Set y-axis limits from 0 to 1
    st.pyplot(fig_combined)
    plt.close(fig_combined) # Close the plot to free up memory