import streamlit as st
import pandas as pd
import numpy as np
import difflib

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)


# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("D:\Movie recommendation\movies (1).csv")

    # Remove movies with no genres listed (you already did it in notebook, still safe here)
    df = df[df["genres"] != "(no genres listed)"].copy()

    df["genres_list"] = df["genres"].str.split("|")
    df["title_lower"] = df["title"].str.lower().str.strip()

    return df


# -----------------------------
# Train Model + Similarity Matrix
# -----------------------------
@st.cache_resource
def train_model(df):
    corpus = df["genres_list"].tolist()

    model = Word2Vec(
        sentences=corpus,
        vector_size=50,
        window=5,
        min_count=1,
        workers=4,
        sg=1
    )

    # Create a vector for each movie (mean of its genre vectors)
    movie_vectors = []
    for genres in corpus:
        vectors = [model.wv[g] for g in genres if g in model.wv]
        movie_vectors.append(np.mean(vectors, axis=0))

    movie_vectors = np.array(movie_vectors)
    sim_matrix = cosine_similarity(movie_vectors)

    return sim_matrix


# -----------------------------
# Helper: Suggest Closest Titles
# -----------------------------
def get_closest_titles(user_input, titles, n=5):
    matches = difflib.get_close_matches(user_input, titles, n=n, cutoff=0.3)
    return matches


# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_movies(df, sim_matrix, selected_title, top_n=10):
    selected_title = selected_title.lower().strip()

    if selected_title not in df["title_lower"].values:
        return None

    idx = df[df["title_lower"] == selected_title].index[0]

    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # skip itself
    top_scores = scores[1:top_n + 1]
    top_indices = [i[0] for i in top_scores]
    similarity_values = [round(s[1], 4) for s in top_scores]

    result = df.loc[top_indices, ["title", "genres"]].copy()
    result["similarity_score"] = similarity_values

    # Better sorting
    result = result.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)

    return result


# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.caption("Content-based recommendation using **Word2Vec (Genres)** + **Cosine Similarity**")

df = load_data()
sim_matrix = train_model(df)

all_titles = df["title"].sort_values().unique().tolist()
all_titles_lower = df["title_lower"].unique().tolist()

st.markdown("---")

# Search input
st.subheader("🔎 Search a Movie")
user_movie_input = st.text_input("Type a movie name (example: Toy Story)", "")

selected_movie = None

if user_movie_input.strip() != "":
    suggestions = get_closest_titles(user_movie_input, all_titles, n=7)

    if suggestions:
        selected_movie = st.selectbox("Did you mean:", suggestions)
    else:
        st.warning("No close matches found. Try a different keyword.")

else:
    selected_movie = st.selectbox("Or select from full list:", all_titles)

top_n = st.slider("🎯 Number of recommendations", min_value=5, max_value=20, value=10)

if st.button("✅ Recommend"):
    if selected_movie is None:
        st.error("Please select a movie first.")
    else:
        # Selected movie details
        selected_row = df[df["title"] == selected_movie].iloc[0]

        st.markdown("### 🎥 Selected Movie")
        st.info(f"**{selected_row['title']}**  \n**Genres:** {selected_row['genres']}")

        results = recommend_movies(df, sim_matrix, selected_movie, top_n=top_n)

        if results is None or results.empty:
            st.error("Movie not found in dataset. Please try another.")
        else:
            st.markdown("### ✅ Recommended Movies")
            st.dataframe(results, use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Word2Vec | Cosine Similarity")
