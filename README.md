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

## Dataset

This project uses the MovieLens dataset. See `dataset/README.txt` for its licence and attribution details.


# VedGrow_ML_03 - 🎬 Movie Recommendation System


The **Movie Recommendation System** is a Machine Learning project developed as **Task 3** during my **VedGrow Machine Learning Internship**. This application recommends movies based on content similarity using the **MovieLens 100K Dataset**.

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

---

## 🚀 Live Demo

🌐 **Streamlit Web App**

**👉 https://your-app-name.streamlit.app**

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

## 🎯 How It Works

1. Open the Streamlit web application.
2. Select a movie from the dropdown menu.
3. Click the **Recommend** button.
4. The system finds movies with similar features.
5. Recommended movies are displayed instantly.

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

> Add screenshot here

```
assets/home.png
```

### Recommendation Results

> Add screenshot here

```
assets/recommendation.png
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
