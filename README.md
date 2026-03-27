# 🎬 Movie Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

A hybrid movie recommendation system combining **collaborative filtering** and **content-based filtering** to suggest personalized movie picks. Built on the MovieLens 25M dataset with a Streamlit frontend.

---

## ✨ Features

- **User-based collaborative filtering** — finds users with similar taste and recommends what they liked
- **Item-based collaborative filtering** — recommends movies similar to ones you've rated highly
- **Content-based filtering** — uses genre, cast, director, and keyword similarity (TF-IDF + cosine similarity)
- **Hybrid scoring** — weighted blend of collaborative and content signals
- **Cold start handling** — content-based fallback for new users with no rating history
- **Interactive Streamlit UI** — search, rate, and get personalized recommendations instantly

---

## 🚀 Quick Start

```bash
git clone https://github.com/TheCromazone/movie-recommender.git
cd movie-recommender

# Install dependencies
pip install -r requirements.txt

# Download MovieLens data (small version for quick testing)
python data/download_data.py --size small

# Build the recommendation index
python build_index.py

# Launch the app
streamlit run app.py
```

---

## 🏗️ Project Structure

```
movie-recommender/
├── app.py                      # Streamlit web application
├── build_index.py              # Precompute similarity matrices
├── recommender/
│   ├── collaborative.py        # User-based and item-based CF
│   ├── content_based.py        # TF-IDF + cosine similarity
│   └── hybrid.py               # Hybrid recommender engine
├── data/
│   ├── download_data.py        # MovieLens dataset downloader
│   └── preprocessing.py        # Data cleaning and preparation
├── requirements.txt
└── README.md
```

---

## 🔬 How It Works

### Collaborative Filtering
Uses matrix factorization (SVD via `surprise`) on the user-item ratings matrix. The model learns latent factors for both users and movies, enabling it to predict ratings for unseen user-movie pairs.

### Content-Based Filtering
Builds a TF-IDF feature matrix from movie metadata (genres, keywords, cast, director). Cosine similarity between movies is used to find the most similar titles.

### Hybrid Blending
```
final_score = α × collaborative_score + (1 - α) × content_score
```
where `α` is tuned per-user based on rating history length.

---

## 📊 Model Performance (MovieLens 25M)

| Method | RMSE | MAE | Precision@10 | Recall@10 |
|---|---|---|---|---|
| User-based CF | 0.921 | 0.711 | 0.34 | 0.22 |
| Item-based CF | 0.887 | 0.682 | 0.41 | 0.28 |
| SVD (matrix factorization) | 0.856 | 0.657 | 0.47 | 0.33 |
| **Hybrid (final)** | **0.831** | **0.634** | **0.53** | **0.39** |

---

## 📦 Dataset

Uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) by GroupLens Research. The small version (100K ratings) is sufficient for local development; the 25M version is recommended for production-quality results.

---

## 🛠️ Tech Stack

- `scikit-surprise` — collaborative filtering (SVD, KNN)
- `scikit-learn` — TF-IDF, cosine similarity
- `pandas` / `numpy` — data manipulation
- `streamlit` — web UI
- `plotly` — visualizations

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
