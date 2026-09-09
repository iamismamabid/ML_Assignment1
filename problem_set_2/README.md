# Problem Set 02: Bank Term Deposit Subscription Prediction

## 1. Project Overview
The objective of this project is to build and train a machine learning model to predict whether a client will subscribe to a bank term deposit based on direct marketing campaign data. By identifying high-propensity leads, the bank can optimize marketing campaigns and allocate outreach resources more effectively.

---

## 2. Dataset & Preprocessing Pipeline
- **Encoding Categorical Features**: Categorical variables (such as job type, marital status, education, contact method) were encoded into numerical representations using one-hot encoding / label encoding.
- **Feature Scaling**: Continuous numerical features were normalized/standardized using `StandardScaler` to bring all input variables onto a uniform scale, facilitating stable gradient descent in neural networks.
- **Data Splitting**: Partitioned the data into training, validation, and testing sets to benchmark model generalization.

---

## 3. Methodology & Model Architecture

### Baseline Model
- **Logistic Regression**: Implemented as a performance baseline to establish the linear floor and assess separability before moving to non-linear neural approaches.

### Deep Learning Architecture (Multilayer Perceptron - MLP)
Implemented using **TensorFlow / Keras**:
- **Input Layer**: Accepts the preprocessed feature vectors.
- **Hidden Layers**: Dense layers equipped with ReLU activation functions.
- **Regularization & Optimization**:
  - Dropout layers and batch normalization to mitigate initial overfitting observed during training.
  - Optimizer: Adam
  - Loss Function: Binary Cross-Entropy
- **Output Layer**: Single dense unit with a Sigmoid activation function outputting the subscription probability.

---

## 4. Evaluation Metrics
Given class imbalance typical of marketing datasets, models were evaluated on:
- **ROC-AUC Score**
- **Precision, Recall, and F1-Score**
- **Confusion Matrix**

---

## 5. Key Findings & Business Impact
- Addressing initial overfitting through regularization and tuning improved out-of-sample validation accuracy and stability.
- The final neural network enables targeted marketing by prioritizing individuals in the top deciles of predicted subscription probability, minimizing contact costs while maximizing conversion rates.
