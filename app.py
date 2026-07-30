from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"

st.set_page_config(page_title="Movie Recommender", layout="wide")


@st.cache_data
def load_movies() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "movies.csv")


@st.cache_data
def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "ratings.csv")


@st.cache_resource
def load_content_artifacts():
    return joblib.load(PROJECT_ROOT / "tfidf_matrix.pkl")


@st.cache_resource
def load_svd_model():
    return joblib.load(PROJECT_ROOT / "svd_model.pkl")


def content_recommendations(
    movie_title: str, movies: pd.DataFrame, tfidf_matrix, limit: int
) -> pd.DataFrame:
    movie_index = movies.index[movies["title"] == movie_title][0]
    scores = cosine_similarity(tfidf_matrix[movie_index],tfidf_matrix).flatten()
    scores = list(enumerate(scores))
    
    scores.sort(key=lambda item: item[1], reverse=True)

    recommendations = movies.iloc[[index for index, _ in scores[1 : limit + 1]]][
        ["title", "genres"]
    ].copy()
    recommendations["Similarity Score"] = [
        round(score, 3) for _, score in scores[1 : limit + 1]
    ]
    return recommendations.reset_index(drop=True)


def property_recommendations(
    movies: pd.DataFrame,
    selected_genres: list[str],
    start_year: int,
    end_year: int,
    match_all_genres: bool,
    limit: int,
) -> pd.DataFrame:
    recommendations = movies.loc[
        movies["year"].between(start_year, end_year, inclusive="both")
    ].copy()

    if selected_genres:
        recommendations["Match Score"] = recommendations["genres"].apply(
            lambda genres: sum(
                genre in genres.split("|") for genre in selected_genres
            )
        )
        recommendations["Matched Genres"] = recommendations["genres"].apply(
            lambda genres: ", ".join(
                genre for genre in selected_genres if genre in genres.split("|")
            )
        )
        required_matches = len(selected_genres) if match_all_genres else 1
        recommendations = recommendations.loc[
            recommendations["Match Score"] >= required_matches
        ]
    else:
        recommendations["Match Score"] = 0
        recommendations["Matched Genres"] = "No genre preference selected"

    recommendations = recommendations.sort_values(
        ["Match Score", "year", "title"], ascending=[False, False, True]
    )
    return recommendations[["title", "genres", "year", "Matched Genres"]].head(limit)


def user_recommendations(
    user_id: int, movies: pd.DataFrame, ratings: pd.DataFrame, svd_model, limit: int
) -> pd.DataFrame:
    watched_ids = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    unseen_movies = movies.loc[~movies["movieId"].isin(watched_ids)]
    predictions = [
        (movie_id, svd_model.predict(user_id, movie_id).est)
        for movie_id in unseen_movies["movieId"]
    ]
    predictions.sort(key=lambda item: item[1], reverse=True)

    result = pd.DataFrame(predictions[:limit], columns=["movieId", "Predicted Rating"])
    result = result.merge(movies, on="movieId")
    result["Predicted Rating"] = result["Predicted Rating"].round(2)
    return result[["title", "genres", "Predicted Rating"]]


movies = load_movies()
ratings = load_ratings()
movies["year"] = pd.to_numeric(
    movies["title"].str.extract(r"\((\d{4})\)$")[0], errors="coerce"
).fillna(0).astype(int)
available_genres = sorted(
    {
        genre
        for genres in movies["genres"]
        for genre in genres.split("|")
        if genre != "(no genres listed)"
    }
)
minimum_year = int(movies.loc[movies["year"] > 0, "year"].min())
maximum_year = int(movies["year"].max())

st.title("Movie Recommendation System")
st.caption("Find films by title, preferred properties, or personalised predictions.")

with st.sidebar:
    st.header("Recommendation Settings")
    mode = st.radio(
        "Choose a method", ["Movie name", "Preferences", "For a user"]
    )
    limit = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if mode == "Movie name":
    search_text = st.text_input(
        "Type a movie name", placeholder="Example: Matrix or Toy Story"
    )
    matching_titles = []
    if len(search_text.strip()) >= 2:
        matching_titles = movies.loc[
            movies["title"].str.contains(search_text.strip(), case=False, na=False), "title"
        ].tolist()

    if search_text and len(search_text.strip()) < 2:
        st.info("Type at least two characters to search for a movie.")
    elif search_text and not matching_titles:
        st.warning("No movie names match your search. Try another title.")
    elif matching_titles:
        selected_title = st.selectbox("Select the movie", matching_titles)
        if st.button("Find similar movies", type="primary"):
            with st.spinner("Loading the content-based model..."):
                tfidf_matrix = load_content_artifacts()
                recommendations = content_recommendations(
                    selected_title, movies, tfidf_matrix, limit
                )
            st.subheader(f"Movies similar to {selected_title}")
            st.dataframe(recommendations, use_container_width=True, hide_index=True)
    else:
        st.info("Enter part or all of a movie name to find similar movies.")

elif mode == "Preferences":
    st.subheader("Choose the movie properties you want")
    selected_genres = st.multiselect("Preferred genres", available_genres)
    start_year, end_year = st.slider(
        "Release year range", minimum_year, maximum_year, (minimum_year, maximum_year)
    )
    match_all_genres = st.checkbox("Require every selected genre", value=False)

    if st.button("Find movies", type="primary"):
        recommendations = property_recommendations(
            movies, selected_genres, start_year, end_year, match_all_genres, limit
        )
        if recommendations.empty:
            st.warning("No movies match those properties. Try broader preferences.")
        else:
            st.subheader("Movies matching your preferences")
            st.dataframe(recommendations, use_container_width=True, hide_index=True)

else:
    available_users = sorted(ratings["userId"].unique())
    user_id = st.selectbox("Choose a user ID", available_users)
    if st.button("Recommend movies", type="primary"):
        with st.spinner("Generating personalised recommendations..."):
            svd_model = load_svd_model()
            recommendations = user_recommendations(user_id, movies, ratings, svd_model, limit)
        st.subheader(f"Top recommendations for user {user_id}")
        st.dataframe(recommendations, use_container_width=True, hide_index=True)

with st.expander("About this project"):
    st.write(
        "Use **Movie name** to find similar titles from genres and user tags. "
        "Use **Preferences** to filter by genres and release years, or **For a user** "
        "for recommendations predicted by the trained SVD model."
    )
