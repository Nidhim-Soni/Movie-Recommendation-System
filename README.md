# 🎬 Movie Recommendation System (Streamlit App)

A **content-based movie recommendation system** that suggests similar movies using  
✅ **Word2Vec (Genre Embeddings)** + ✅ **Cosine Similarity**.

---

## 🚀 Live Demo
👉 Deployed on Streamlit: (https://movie-recommendation-system-nidhim-soni.streamlit.app/)

---

## 📌 Project Overview
Recommendation systems are widely used in platforms like **Netflix, Prime Video, YouTube**, etc.  
This project recommends movies based on **genre similarity** by learning embeddings using **Word2Vec**.

✅ Input: A movie title  
✅ Output: Top-N similar movie recommendations

---

## 🧠 Approach Used
### ✅ Content-Based Filtering
- Extracted genres from the dataset  
- Converted genres into embeddings using **Word2Vec (Skip-Gram)**
- Created a **movie vector** using the mean of its genre vectors
- Used **cosine similarity** to find closest movies

---

## 📂 Dataset
- Dataset: MovieLens metadata (`movies.csv`)
- Columns used:
  - `movieId`
  - `title`
  - `genres`

✅ Movies with `(no genres listed)` were removed during preprocessing.

---

## 🎯 Features
✅ Search & select movies easily  
✅ Choose number of recommendations (Top-N)  
✅ Shows recommended movies with genres  
✅ Displays similarity score for transparency  
✅ Clean and interactive Streamlit UI

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Gensim (Word2Vec)
- Scikit-learn (Cosine Similarity)
- Streamlit (Deployment)

---

## ▶️ How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Movie-Recommendation-System.git
cd Movie-Recommendation-System

