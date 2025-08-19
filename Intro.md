# Neural Networks

---

## 1. Introduction to Neural Networks

A **Neural Network (NN)** is a machine learning model inspired by the biological brain. It is designed to recognize patterns and learn mappings from input data (features) to output predictions (labels or continuous values). Unlike traditional algorithms where the features are hand-engineered, neural networks can **automatically learn representations** from data.

At a high level, a neural network is a collection of layers where:
- The **input layer** receives the raw data.
- The **hidden layers** transform the input into meaningful intermediate representations.
- The **output layer** generates the prediction (e.g., classification, regression, or generative output).

Each connection between neurons has an associated **weight**, which determines the strength of the influence between two neurons. During training, the network adjusts these weights to minimize an **error function** (loss).

---

## 2. Biological Inspiration

Neural networks are inspired by the **human brain’s neural structure**.  
- A biological **neuron** has:
  - **Dendrites**: receive electrical signals (inputs).
  - **Cell Body (Soma)**: processes the signals.
  - **Axon**: transmits the output signal to other neurons.

In an **artificial neural network**:
- **Input values** correspond to dendrites.
- A **weighted sum + bias** mimics the signal processing inside the soma.
- An **activation function** decides whether the neuron “fires” (produces output).
- The **output** corresponds to the axon signal.

This analogy provides the motivation for the mathematical design of artificial neurons.

---

## 3. The Perceptron: The Building Block of NNs

The perceptron, introduced by **Frank Rosenblatt (1958)**, is the simplest form of a neural network. It is essentially a single-layer linear classifier.

### 3.1 Mathematical Model of Perceptron

Given input features:
\[
x = [x_1, x_2, ..., x_n]
\]

And corresponding weights:
\[
w = [w_1, w_2, ..., w_n]
\]

The perceptron computes a weighted sum:
\[
z = \sum_{i=1}^{n} w_i x_i + b
\]

Then applies an **activation function** \( f(z) \):
\[
y = f(z)
\]

where:
- \( b \) is the bias term, shifting the decision boundary.
- \( y \) is the final output.

### 3.2 Training the Perceptron
Training involves adjusting weights using the **perceptron learning rule**:
\[
w_i^{new} = w_i^{old} + \eta (y_{true} - y_{pred}) x_i
\]

where:
- \( \eta \) = learning rate.
- \( y_{true} \) = actual label.
- \( y_{pred} \) = perceptron’s output.

### 3.3 Limitations
- The perceptron can only solve **linearly separable problems** (e.g., AND, OR).
- It **fails** on non-linear problems like XOR because a single line cannot separate the classes.
- This limitation led to the development of **Multi-Layer Perceptrons (MLPs)** with non-linear activations.

---

## 4. Activation Functions

Activation functions introduce **non-linearity** into the network, enabling it to learn **complex decision boundaries**.

### 4.1 Step Function
- **Definition**: Outputs `1` if input > threshold, otherwise `0`.
- **Limitation**: Non-differentiable, unsuitable for gradient-based learning.

### 4.2 Sigmoid Function
\[
\sigma(x) = \frac{1}{1+e^{-x}}
\]

- Output in range (0,1).
- Interpretable as probability.
- **Problems**:
  - **Vanishing gradient** for very large/small inputs.
  - Slow convergence.

### 4.3 Hyperbolic Tangent (tanh)
\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

- Range: (-1, 1).
- Centered at zero (better than sigmoid).
- Still suffers from vanishing gradients.

### 4.4 ReLU (Rectified Linear Unit)
\[
f(x) = \max(0, x)
\]

- Very efficient, widely used in deep learning.
- **Problems**:
  - Dead neurons (when weights become negative, gradient = 0).
- Variants: Leaky ReLU, Parametric ReLU, ELU.

### 4.5 Softmax Function
- Converts raw scores into probabilities across multiple classes.
\[
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

Used in **multi-class classification tasks**.

---

## 5. Neural Network Architecture

### 5.1 Layers
- **Input Layer**: Holds raw features.
- **Hidden Layers**: Perform transformations and feature extraction.
- **Output Layer**: Produces the prediction.

### 5.2 Dense Layer Operation
Each neuron in one layer is connected to **every neuron** in the next layer.  

The process is:
\[
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
\]
\[
a^{(l)} = f(z^{(l)})
\]

where:
- \( W^{(l)} \) = weight matrix of layer \( l \).
- \( b^{(l)} \) = bias vector.
- \( a^{(l)} \) = activation output.

---

## 6. Forward Propagation

The mechanism by which input flows through the network:
1. Multiply inputs by weights and add bias.
2. Apply activation functions.
3. Pass results to the next layer.
4. Continue until the output layer.

Forward propagation is essentially **computing predictions**.

---

## 7. Loss Functions

Loss functions measure how far predictions are from the actual values.

### 7.1 Mean Squared Error (MSE)
\[
L = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
\]
- Used for regression.

### 7.2 Cross-Entropy Loss
For binary classification:
\[
L = - \left[y \log(\hat{y}) + (1-y)\log(1-\hat{y})\right]
\]

For multi-class classification:
\[
L = - \sum_{i=1}^n y_i \log(\hat{y}_i)
\]

---

## 8. Backpropagation

Backpropagation is the **learning algorithm** used in NNs to update weights.

### 8.1 Chain Rule
If output depends on intermediate variables:
\[
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
\]

### 8.2 Weight Update
Using Gradient Descent:
\[
w = w - \eta \frac{\partial L}{\partial w}
\]

---

## 9. Gradient Descent and Optimizers

### 9.1 Variants
- **Batch GD**: Updates after processing the whole dataset.
- **Stochastic GD**: Updates after every sample.
- **Mini-Batch GD**: Updates after small batches.

### 9.2 Optimizers
- **SGD**: Simple gradient descent.
- **Adam**: Combines momentum and adaptive learning rates.
- **RMSProp**: Scales learning rate by squared gradients.
- **Adagrad**: Adapt learning rate for each parameter.

---

## 10. Regularization

Overfitting occurs when the model memorizes training data but fails on new data. Regularization prevents this.

- **L1 Regularization (Lasso):** Encourages sparsity.
- **L2 Regularization (Ridge):** Encourages small weights.
- **Dropout:** Randomly disables neurons during training.
- **Batch Normalization:** Normalizes activations in each layer.
- **Early Stopping:** Stops training when validation error increases.

---

## 11. Types of Neural Networks

### 11.1 Feedforward Neural Networks (FNNs)
- Information flows in one direction (input → output).

### 11.2 Convolutional Neural Networks (CNNs)
- Specialized for images.
- Use **convolutional filters** to extract local features.

### 11.3 Recurrent Neural Networks (RNNs)
- Handle sequential data (time series, text).
- Maintain memory of past inputs.

### 11.4 LSTMs and GRUs
- Solve vanishing gradient problem in RNNs.
- LSTMs use **gates** to control memory retention.

### 11.5 Autoencoders
- Learn compressed representations of data.

### 11.6 Generative Adversarial Networks (GANs)
- **Generator** creates fake data.
- **Discriminator** tries to distinguish real vs fake.
- Both trained adversarially.

### 11.7 Transformers
- Based on **attention mechanism**.
- Do not rely on recurrence, enabling parallelization.
- Foundation for models like BERT, GPT.

---

## 12. Advanced Topics

### 12.1 Transfer Learning
- Using pre-trained models on new tasks (e.g., ResNet on ImageNet applied to medical images).

### 12.2 Attention Mechanism
- Assigns **different importance (weights)** to different parts of input sequences.

### 12.3 Residual Networks (ResNet)
- Introduce **skip connections** to train very deep networks.

### 12.4 Capsule Networks
- Preserve spatial relationships between features.

### 12.5 Self-Supervised Learning
- Models learn representations without labeled data (e.g., predicting missing words).

---

## 13. Applications of Neural Networks

- **Computer Vision**: Object detection, medical imaging, autonomous driving.
- **Natural Language Processing**: Machine translation, chatbots, summarization.
- **Finance**: Fraud detection, stock prediction.
- **Healthcare**: Disease diagnosis, drug discovery.
- **Generative AI**: Image generation, music composition, deepfake creation.

---

## 14. Practical Implementation (Workflow)

1. **Problem Definition** (classification, regression, etc.)
2. **Data Collection & Preprocessing** (cleaning, normalization, feature engineering)
3. **Model Architecture Design** (layers, activation functions, etc.)
4. **Choosing Loss & Optimizer**
5. **Training with Forward + Backpropagation**
6. **Hyperparameter Tuning**
7. **Regularization**
8. **Model Evaluation (Accuracy, F1, AUC, etc.)**
9. **Deployment & Monitoring**

---

## 15. Conclusion

Neural networks have evolved from **simple perceptrons** to **transformers powering generative AI**.  
- They are **universal function approximators**, meaning they can theoretically model any function.  
- The main challenges remain in **efficiency, interpretability, fairness, and safety**.  
- Modern research focuses on making them more **scalable, energy-efficient, and trustworthy**.  

---
