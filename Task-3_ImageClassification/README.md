# CNN Image Classification with CIFAR-10

## Overview
This project implements **Convolutional Neural Networks (CNNs)** to classify images from the **CIFAR-10 dataset** into 10 different categories using **TensorFlow and Keras**.

## Dataset
- **CIFAR-10 Dataset** (loaded via `tensorflow.keras.datasets.cifar10`).
- Contains **60,000 images** in 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
- Each image is **32x32 pixels** with **3 color channels (RGB)**.

## Implementation Steps
1. **Load the Dataset**: Split into training and test sets.
2. **Preprocess Data**:
   - Normalize pixel values to [0,1].
   - Convert class labels to **one-hot encoding**.
3. **Build the CNN Model**:
   - 3 convolutional layers with **ReLU activation**.
   - MaxPooling layers to reduce dimensions.
   - Fully connected dense layers.
   - Output layer with **softmax activation** for multi-class classification.
4. **Compile the Model**:
   - **Optimizer**: Adam
   - **Loss Function**: Categorical Cross-Entropy
   - **Metrics**: Accuracy
5. **Train the Model**:
   - **Epochs**: 10
   - **Validation Data**: Test Set
6. **Evaluate Performance**:
   - **Test Accuracy** printed after evaluation.
   - **Training and validation accuracy/loss curves** plotted.
7. **Save the Trained Model**:
   - The trained model is saved as `cnn_image_classification.h5`.

## Files in This Repository
- `cnn_image_classification.ipynb`: Jupyter Notebook with full implementation.
- `cnn_image_classification.h5`: Saved trained CNN model.
- `README.md`: Documentation for the project.

## How to Run
1. Install dependencies:
   ```sh
   pip install tensorflow numpy matplotlib
   ```
2. Run the Python script or Jupyter Notebook:
   ```sh
   python cnn_image_classification.py
   ```
3. Example usage:
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model("cnn_image_classification.h5")
   ```

## Results
- The trained CNN achieves **high accuracy** on test data.
- Performance is visualized using training **accuracy/loss curves**.

## Future Enhancements
- **Data Augmentation** to improve generalization.
- **More CNN Layers** for improved accuracy.
- **Hyperparameter Tuning** to optimize learning.

---
### Author
[Your Name]  
Date: [Current Date]

