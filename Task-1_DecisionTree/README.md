# Decision Tree Classification

## Overview
This project builds a **Decision Tree Classifier** using the **Iris dataset**. The model is trained to classify iris flowers into three species based on their features.

## Dataset
- **Iris dataset** (from `sklearn.datasets.load_iris`)
- Features: Sepal length, Sepal width, Petal length, Petal width
- Target classes: Setosa, Versicolor, Virginica

## Implementation Steps
1. **Load the Dataset**: Extract features and target labels.
2. **Split the Data**: 70% training, 30% testing.
3. **Train a Decision Tree Model**: Using `DecisionTreeClassifier` from Scikit-Learn.
4. **Evaluate Model Performance**:
   - Accuracy Score
   - Classification Report
   - Confusion Matrix
5. **Visualize the Model**:
   - Heatmap of Confusion Matrix
   - Decision Tree Diagram
6. **Save the Model**: The trained model is saved as `decision_tree_model.pkl`.

## Files in This Repository
- `decision_tree.ipynb`: Jupyter Notebook with full implementation.
- `decision_tree_model.pkl`: Saved trained model.
- `README.md`: Documentation for the project.

## How to Run
1. Install dependencies:
   ```sh
   pip install numpy matplotlib seaborn scikit-learn joblib
   ```
2. Run the Python script or Jupyter Notebook:
   ```sh
   python decision_tree.py
   ```
3. Example usage:
   ```python
   import joblib
   model = joblib.load("decision_tree_model.pkl")
   predictions = model.predict([[5.1, 3.5, 1.4, 0.2]])
   print("Predicted class:", predictions)
   ```

## Results
- Achieved **high accuracy** on test data.
- Model is visualized using a **decision tree diagram**.

## Future Enhancements
- Implement **hyperparameter tuning** to optimize tree depth.
- Try **Random Forest Classifier** for better performance.

---
### Author
Abhishek Pandey 
Date: 23-02-2025

