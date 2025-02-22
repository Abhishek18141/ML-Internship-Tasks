# Movie Recommendation System (User-Based Collaborative Filtering)

## Overview

This project builds a **Movie Recommendation System** using **User-Based Collaborative Filtering**. It computes user similarity using **cosine similarity** and recommends movies based on similar users' preferences.

## Dataset

- **ratings.csv**: Contains user ratings for different movies.
- **movies.csv**: Contains movie metadata (movieId and title).

## Implementation Details

- **Data Preprocessing**:
  - Merged ratings with movie titles.
  - Aggregated duplicate ratings by taking the mean.
  - Created a **User-Item matrix** with users as rows and movie titles as columns.
  - Filled missing values with 0.
- **Model Building**:
  - Converted the User-Item matrix into a sparse matrix.
  - Computed **cosine similarity** between users.
  - Stored user similarity in a **user\_similarity\_model.pkl** file.
  - Stored the User-Item matrix in a **user\_item\_matrix.pkl** file.
- **Movie Recommendation**:
  - Identifies similar users based on cosine similarity.
  - Suggests movies that the target user has not seen but similar users have rated.

## Files in This Repository

- `recommendation_system.ipynb`: Jupyter Notebook with full implementation.
- `ratings.csv`: Movie ratings dataset.
- `movies.csv`: Movie metadata.
- `user_similarity_model.pkl`: Saved similarity model.
- `user_item_matrix.pkl`: Saved User-Item matrix.
- `README.md`: Documentation for the project.

## How to Run

1. Install required dependencies:
   ```sh
   pip install pandas numpy scikit-learn scipy joblib
   ```
2. Run the Python script or Jupyter Notebook:
   ```sh
   python recommendation_system.py
   ```
3. Example usage:
   ```python
   recommend_movies(user_id=1, n=5)
   ```

## Results

The system recommends movies to users based on their similarity to others. Results can be evaluated using qualitative analysis of recommendations.

## Future Enhancements

- Implement **Item-Based Collaborative Filtering**.
- Use **Matrix Factorization (SVD, NMF)** for better recommendations.
- Deploy as a web app using **Flask** or **Streamlit**.

---

### Author

Abhishek Pandey\
Date:23-02-2025

