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

---

## 3. The Perceptron: The Building Block of NNs

The perceptron, introduced by **Frank Rosenblatt (1958)**, is the simplest form of a neural network. It is essentially a single-layer linear classifier.

### 3.1 Mathematical Model of Perceptron

Given input features:  
`x = [x1, x2, ..., xn]`

And corresponding weights:  
`w = [w1, w2, ..., wn]`

The perceptron computes a weighted sum:  
`z = (w1*x1 + w2*x2 + ... + wn*xn) + b`

Then applies an **activation function f(z)**:  
`y = f(z)`

where:
- `b` is the bias term.
- `y` is the final output.

### 3.2 Training the Perceptron
The perceptron learning rule:  
`wi(new) = wi(old) + η * (y_true - y_pred) * xi`

where:
- `η` = learning rate.
- `y_true` = actual label.
- `y_pred` = perceptron’s output.

### 3.3 Limitations
- Can only solve **linearly separable problems** (e.g., AND, OR).
- **Fails** on non-linear problems like XOR.
- Led to the development of **Multi-Layer Perceptrons (MLPs)**.

---

## 4. Activation Functions

Activation functions introduce **non-linearity** into the network.

### 4.1 Step Function
Outputs `1` if input > threshold, else `0`.  
- **Limitation**: Not differentiable.

### 4.2 Sigmoid Function
`sigma(x) = 1 / (1 + e^(-x))`  
- Range: (0,1)  
- Problems: vanishing gradients.

### 4.3 Hyperbolic Tangent (tanh)
`tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`  
- Range: (-1,1)  
- Better than sigmoid but still suffers vanishing gradients.

### 4.4 ReLU (Rectified Linear Unit)
`ReLU(x) = max(0, x)`  
- Very efficient, widely used.  
- Variants: Leaky ReLU, Parametric ReLU, ELU.

### 4.5 Softmax Function
For multi-class classification:  
`softmax(zi) = e^(zi) / Σ e^(zj)`

---

## 5. Neural Network Architecture

### 5.1 Layers
- **Input Layer**: raw features.  
- **Hidden Layers**: intermediate processing.  
- **Output Layer**: final predictions.

### 5.2 Dense Layer Operation
For layer `l`:  
`z(l) = W(l) * a(l-1) + b(l)`  
`a(l) = f(z(l))`

where:
- `W(l)` = weight matrix of layer l  
- `b(l)` = bias vector  
- `a(l)` = activation output  

---

## 6. Forward Propagation

Steps:
1. Multiply inputs by weights and add bias.  
2. Apply activation functions.  
3. Pass results to next layer.  
4. Continue until the output layer.  

This is essentially computing predictions.

---

## 7. Loss Functions

Loss functions measure the error.

### 7.1 Mean Squared Error (MSE)
`L = (1/n) * Σ (yi - y_hat_i)^2`  
Used for regression.

### 7.2 Cross-Entropy Loss
Binary classification:  
`L = -(y*log(y_hat) + (1-y)*log(1-y_hat))`

Multi-class classification:  
`L = - Σ yi * log(y_hat_i)`

---

## 8. Backpropagation

Backpropagation updates weights using derivatives.

### 8.1 Chain Rule
If output depends on intermediate variables:  
`dL/dw = (dL/da) * (da/dz) * (dz/dw)`

### 8.2 Weight Update
`w = w - η * dL/dw`

---

## 9. Gradient Descent and Optimizers

### 9.1 Variants
- **Batch GD**: Updates once per epoch (all data).  
- **Stochastic GD**: Updates after every sample.  
- **Mini-Batch GD**: Updates after a small batch.  

### 9.2 Optimizers
- **SGD**: Simple gradient descent.  
- **Adam**: Combines momentum + adaptive learning rates.  
- **RMSProp**: Uses squared gradient scaling.  
- **Adagrad**: Adapts learning rates per parameter.  

---

## 10. Regularization

To prevent overfitting:
- **L1 Regularization**: encourages sparsity.  
- **L2 Regularization**: penalizes large weights.  
- **Dropout**: randomly disables neurons.  
- **Batch Normalization**: normalizes layer inputs.  
- **Early Stopping**: stops training when validation error rises.  

---

## 11. Types of Neural Networks

- **Feedforward NN (FNN)**: data flows forward only.  
- **Convolutional NN (CNN)**: image feature extraction.  
- **Recurrent NN (RNN)**: sequential data.  
- **LSTMs/GRUs**: improved RNNs for long-term memory.  
- **Autoencoders**: dimensionality reduction.  
- **GANs**: generator + discriminator.  
- **Transformers**: self-attention models, foundation of GPT/BERT.  

---

## 12. Advanced Topics

- **Transfer Learning**: reuse pre-trained models.  
- **Attention Mechanism**: assign importance to inputs.  
- **ResNets**: skip connections for deep networks.  
- **Capsule Networks**: preserve spatial hierarchies.  
- **Self-Supervised Learning**: learns without explicit labels.  

---

## 13. Applications

- Computer Vision (object detection, medical imaging)  
- NLP (chatbots, translation, summarization)  
- Finance (fraud detection, trading)  
- Healthcare (diagnosis, drug discovery)  
- Generative AI (text, music, images)  

---

## 14. Workflow for Building a Neural Network

1. Define problem (classification/regression).  
2. Collect and preprocess data.  
3. Design architecture.  
4. Select loss and optimizer.  
5. Train using forward + backpropagation.  
6. Tune hyperparameters.  
7. Apply regularization.  
8. Evaluate performance.  
9. Deploy and monitor.  

---

## 15. Conclusion

Neural networks evolved from simple **perceptrons** to advanced **transformers**.  
They are universal function approximators but face challenges like **efficiency, interpretability, and fairness**.  
Research is ongoing to make them more **scalable, energy-efficient, and trustworthy**.  

---
