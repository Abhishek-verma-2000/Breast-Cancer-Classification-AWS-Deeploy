# Breast-Cancer-Classification-AWS-Deeploy

Breast Cancer Classification using Deep Learning

Project Overview:
This project implements a deep learning model to classify breast cancer images as either benign or malignant. The model utilizes transfer learning with a pre-trained DenseNet201 architecture, achieving high accuracy in distinguishing between the two classes.

Key Technologies and Libraries:
1. Python 3.x
2. TensorFlow 2.x and Keras for deep learning
3. OpenCV and Pillow for image processing
4. NumPy and Pandas for data manipulation
5. Scikit-learn for data splitting and metrics
6. Matplotlib for visualization

Data Preparation:
- Dataset: 239 breast cancer images (112 benign, 127 malignant)
- Image size: Resized to 224x224 pixels
- Data augmentation: Implemented using Keras ImageDataGenerator
  - Techniques: Random zoom, rotation, horizontal and vertical flips

Model Architecture:
- Base model: DenseNet201 (pre-trained on ImageNet)
- Additional layers:
  - Global Average Pooling
  - Dropout (rate: 0.5)
  - Batch Normalization
  - Dense output layer (2 units, softmax activation)
- Total parameters: 18,333,506
  - Trainable: 18,100,610
  - Non-trainable: 232,896

Training Details:
- Optimizer: Adam (learning rate: 1e-4)
- Loss function: Binary Cross-Entropy
- Batch size: 16
- Epochs: 7
- Learning rate reduction: ReduceLROnPlateau callback
- Data split: 80% training, 20% validation

Performance Metrics:
- Final training accuracy: 100%
- Final validation accuracy: 91.67%
- Final training loss: 0.0159
- Final validation loss: 0.2203

Training Progress:
- Epoch 1: Val Accuracy: 72.92%, Val Loss: 0.4609
- Epoch 3: Val Accuracy: 89.58%, Val Loss: 0.2878
- Epoch 7: Val Accuracy: 91.67%, Val Loss: 0.2203

Model Evaluation:
The model shows excellent performance on the training set, achieving 100% accuracy. The validation accuracy of 91.67% indicates good generalization to unseen data. However, the gap between training and validation metrics suggests some overfitting, which could potentially be addressed with more regularization or a larger dataset.

Conclusion:
This project demonstrates proficiency in:
1. Implementing transfer learning with state-of-the-art CNN architectures
2. Handling imbalanced medical imaging datasets
3. Applying data augmentation techniques
4. Fine-tuning deep learning models for binary classification tasks
5. Utilizing TensorFlow and Keras for efficient model development
6. Implementing callbacks for learning rate adjustment and model checkpointing

The high accuracy achieved on a relatively small dataset showcases the potential of this approach for medical image classification tasks. Future work could focus on collecting more data, experimenting with other architectures, and implementing additional regularization techniques to further improve model generalization.
