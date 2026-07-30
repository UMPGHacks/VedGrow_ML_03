# VedGrow_ML_03 - 🎬 Movie Recommendation System

An end-to-end **Machine Learning Movie Recommendation System** developed as **Task 3** during my **VedGrow Machine Learning Internship**.
This application recommends movies based on content similarity using the **MovieLens 100K Dataset**. 

The project implements a **Content-Based Recommendation System** that analyzes movie features and suggests similar movies. It also includes a **Streamlit web application** for an interactive user experience.

---

## ✨ Features

- 🎥 Content-Based Movie Recommendation
- 📊 Data preprocessing and cleaning
- 🔍 Similarity-based movie suggestions
- ⚡ Fast recommendation using Cosine Similarity
- 🌐 Interactive Streamlit web application
- 💻 Simple and user-friendly interface

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle
- MovieLens 100K Dataset

---

## 📂 Project Structure

```text
movie-recommendation-system/
├── app.py                         # Streamlit application
├── movie_recomendation.ipynb      # Data analysis, training, and evaluation
├── requirements.txt               # Python dependencies
├── .gitignore
├── README.md
├── dataset/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── links.csv
├── cosine_similarity.pkl          # Content-based model artifact
├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
├── svd_model.pkl                  # Collaborative-filtering model
├── knn_model.pkl                  # KNN baseline model
└── evaluation_results.csv
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/VedGrow_ML_03.git
```

### 2. Navigate to the project directory

```bash
cd VedGrow_ML_03
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## Models

| Method | Purpose |
| --- | --- |
| TF-IDF + cosine similarity | Finds movies with similar genres and tags. |
| SVD | Predicts ratings for movies a user has not watched. |
| KNNBasic | Collaborative-filtering baseline evaluated in the notebook. |

## Evaluation

| Metric | Value |
| --- | ---: |
| RMSE | 0.8807 |
| MAE | 0.6766 |
| Precision@10 | 0.7446 |
| Recall@10 | 0.8070 |

---

## 🚀 Live Demo

🌐 **Streamlit Web App**

**👉 [https://your-app-name.streamlit.app](https://movie-recomendation-system-ufyxjjgbw24qk85efan88j.streamlit.app/)**

---

## 📊 Machine Learning Workflow

- Data Collection
- Data Cleaning
- Feature Engineering
- Content-Based Filtering
- Cosine Similarity Calculation
- Movie Recommendation
- Streamlit Deployment

---

## ⚙️ How It Works

The Movie Recommendation System provides recommendations using three different approaches:

### 🎬 1. Content-Based Recommendation

- The user searches and selects a movie title.
- The movie's genres and textual features are converted into numerical vectors using **TF-IDF Vectorization**.
- **Cosine Similarity** is used to compare the selected movie with all other movies in the dataset.
- Movies with the highest similarity scores are returned as recommendations.

---

### 🎯 2. Preference-Based Recommendation

- Users choose their preferred genres and release year range.
- The system filters movies that match the selected preferences.
- Results are ranked based on genre matches and release year.

---

### 👤 3. Personalized Recommendation

- The user selects a User ID from the MovieLens dataset.
- A trained **SVD (Singular Value Decomposition)** collaborative filtering model predicts ratings for movies the user has not watched.
- Movies with the highest predicted ratings are recommended.

---

### 🌐 Web Application Workflow

1. Launch the Streamlit web application.
2. Choose one of the three recommendation methods.
3. Provide the required input (movie title, preferences, or user ID).
4. Click the recommendation button.
5. The ML model processes the request and generates personalized movie recommendations instantly.
---

## 📈 Future Improvements

- Collaborative Filtering
- Hybrid Recommendation System
- TMDB API Integration
- Movie Posters
- Personalized User Recommendations
- User Login System

---

## 📸 Screenshots

### Home Page

```
web app screenshot\home.png
```

### Recommendation Results

```
web app screenshot\prediction.png
```

---

## 📋 Requirements

Install all required libraries using:

```bash
pip install -r requirements.txt
```

---

## 🤝 Internship

This project was developed as part of the **VedGrow Machine Learning Internship (Task 3)** to gain practical experience in building and deploying recommendation systems using Machine Learning.

---

## 👨‍💻 Author

**Parag Gupta**

- GitHub: https://github.com/UMPGHacks
- LinkedIn: https://www.linkedin.com/in/parag-gupta-89941828a/

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub. It helps support the project and motivates future improvements.

---
