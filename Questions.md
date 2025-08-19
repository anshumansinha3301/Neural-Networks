#Questions 

---

## Q1. What is a Neural Network and why do we use it?

**Answer:**  
A Neural Network is a computational model inspired by the human brain. It consists of interconnected nodes (neurons) arranged in layers (input, hidden, and output). Each connection between neurons has an associated **weight** that determines how strongly the input affects the output.  

We use Neural Networks because:
- They can approximate any continuous function (Universal Approximation Theorem).
- They automatically learn **features** from data without manual engineering.
- They work well for tasks where classical algorithms fail, such as image recognition, speech processing, and natural language understanding.

---

## Q2. Explain the structure of a single neuron (Perceptron).

**Answer:**  
A perceptron receives multiple inputs `x1, x2, ..., xn`, each multiplied by a weight `w1, w2, ..., wn`. A bias `b` is added to shift the decision boundary.  
The neuron then computes:

`z = (w1*x1 + w2*x2 + ... + wn*xn) + b`  

This value is passed through an **activation function** `f(z)` to produce the final output `y`.  

- Inputs = features (e.g., pixel intensities).  
- Weights = importance assigned to each feature.  
- Bias = helps shift decision boundary.  
- Activation = decides if the neuron "fires".  

---

## Q3. Why are activation functions important?

**Answer:**  
Without activation functions, a neural network is just a **linear function** regardless of depth. This limits it to learning only linearly separable patterns.  

Activation functions introduce **non-linearity**, enabling networks to model complex decision boundaries. Examples:  
- **Sigmoid:** smooth, used in probability outputs.  
- **Tanh:** zero-centered, better for hidden layers.  
- **ReLU:** fast, widely used in deep networks.  
- **Softmax:** converts raw outputs into class probabilities.  

Thus, activation functions make deep learning possible.

---

## Q4. What is the difference between Single-Layer and Multi-Layer Perceptrons?

**Answer:**  
- **Single-Layer Perceptron (SLP):**  
  - Consists of only input and output layer.  
  - Can only solve **linearly separable problems** (e.g., AND, OR).  
  - Cannot solve XOR problem.  

- **Multi-Layer Perceptron (MLP):**  
  - Contains one or more hidden layers.  
  - Uses non-linear activation functions.  
  - Can approximate **non-linear functions** and solve complex tasks.  

---

## Q5. What is Forward Propagation?

**Answer:**  
Forward Propagation is the process of computing the **output of a neural network given an input**.  

Steps:
1. Input is multiplied by weights and added with bias.  
2. Result is passed through an activation function.  
3. Output from one layer becomes input to the next.  
4. Final output layer produces prediction (classification or regression).  

Example for layer `l`:  
`z(l) = W(l) * a(l-1) + b(l)`  
`a(l) = f(z(l))`  

where `a(l-1)` is the previous layer’s output.

---

## Q6. What is Backpropagation and why is it needed?

**Answer:**  
Backpropagation is the **training algorithm** for neural networks. It computes how much each weight contributes to the error and updates weights accordingly.  

Steps:
1. Compute prediction via forward pass.  
2. Calculate loss using a loss function.  
3. Apply **chain rule** of calculus to compute partial derivatives of loss with respect to weights.  
4. Update weights using gradient descent:  
   `w = w - η * dL/dw`  

Without backpropagation, we wouldn’t know how to adjust weights to reduce error.

---

## Q7. Explain Gradient Descent and its variants.

**Answer:**  
Gradient Descent is an optimization algorithm that updates parameters in the opposite direction of the gradient of the loss function.  

Types:
- **Batch Gradient Descent:** Uses entire dataset; stable but slow.  
- **Stochastic Gradient Descent (SGD):** Updates after each sample; faster but noisy.  
- **Mini-Batch Gradient Descent:** Trade-off; updates after small batches.  

Variants (optimizers):
- **Momentum:** Helps escape local minima.  
- **Adam:** Adaptive learning rate + momentum.  
- **RMSProp:** Scales learning rate by squared gradient.  

---

## Q8. What are Loss Functions? Give examples.

**Answer:**  
Loss functions measure how far predictions are from the true values.  

- **Regression:**  
  - Mean Squared Error (MSE) = `(1/n) * Σ (yi - y_hat_i)^2`  
- **Binary Classification:**  
  - Binary Cross-Entropy = `-(y*log(y_hat) + (1-y)*log(1-y_hat))`  
- **Multi-Class Classification:**  
  - Categorical Cross-Entropy = `-Σ yi * log(y_hat_i)`  

The goal of training is to **minimize loss**.

---

## Q9. What is the Vanishing Gradient Problem?

**Answer:**  
In very deep networks, during backpropagation, gradients become extremely small as they are multiplied layer by layer (especially with sigmoid/tanh activations). This causes:
- Very slow or no learning.  
- Weights stop updating.  

**Solutions:**
- Use ReLU instead of sigmoid/tanh.  
- Batch normalization.  
- Skip connections (ResNet).  

---

## Q10. Explain Overfitting in Neural Networks.

**Answer:**  
Overfitting happens when a model memorizes training data instead of generalizing to unseen data. Symptoms include:
- High training accuracy but low validation/test accuracy.  

**Techniques to reduce overfitting:**
- Regularization (L1, L2).  
- Dropout (randomly disabling neurons).  
- Early stopping (stop when validation loss rises).  
- Data augmentation.  

---

## Q11. What is Dropout and why is it used?

**Answer:**  
Dropout randomly "drops" (sets to zero) a percentage of neurons during training. This prevents neurons from relying too much on each other (co-adaptation).  

For example, with dropout rate = 0.5, half of the neurons are disabled randomly at each training step.  

**Benefits:**  
- Prevents overfitting.  
- Forces the network to learn **robust features**.  

---

## Q12. Explain Batch Normalization.

**Answer:**  
Batch Normalization normalizes the output of each layer so that it has **zero mean and unit variance**.  

Steps:
1. Compute mean and variance of activations in a batch.  
2. Normalize: `(x - mean) / sqrt(variance + epsilon)`  
3. Scale and shift using learnable parameters.  

**Advantages:**  
- Faster convergence.  
- Reduces vanishing gradient.  
- Acts as a regularizer.  

---

## Q13. What are Convolutional Neural Networks (CNNs)?

**Answer:**  
CNNs are specialized neural networks for **image and spatial data**. Instead of connecting every neuron, CNNs use **convolutional filters** to detect features like edges, textures, and shapes.  

Key components:
- **Convolutional Layer:** applies filters over input.  
- **Pooling Layer:** reduces spatial size (max/average pooling).  
- **Fully Connected Layer:** final decision-making.  

CNNs are widely used in object detection, facial recognition, and self-driving cars.

---

## Q14. What are Recurrent Neural Networks (RNNs)?

**Answer:**  
RNNs are designed for **sequential data** (time-series, text, speech). Unlike feedforward networks, RNNs have **feedback connections**, allowing them to maintain memory of past inputs.  

- They process inputs step by step while keeping a hidden state.  
- Problems: suffer from vanishing gradient for long sequences.  

---

## Q15. What are LSTMs and GRUs?

**Answer:**  
- **LSTM (Long Short-Term Memory):**  
  Uses input, forget, and output gates to control memory flow. Helps retain long-term dependencies.  

- **GRU (Gated Recurrent Unit):**  
  Simplified version of LSTM with fewer gates (reset and update gates). Faster and performs well on smaller datasets.  

Both solve the vanishing gradient problem in RNNs.

---

## Q16. What are Autoencoders?

**Answer:**  
Autoencoders are neural networks trained to **reconstruct their input**.  

Architecture:
- **Encoder:** compresses input to lower-dimensional latent representation.  
- **Decoder:** reconstructs input from latent space.  

Applications:
- Dimensionality reduction.  
- Denoising.  
- Anomaly detection.  

---

## Q17. Explain Generative Adversarial Networks (GANs).

**Answer:**  
GANs consist of two networks:  
1. **Generator (G):** Produces fake samples from random noise.  
2. **Discriminator (D):** Tries to distinguish real samples from fake ones.  

Training is adversarial:
- G improves to fool D.  
- D improves to detect fakes.  

Applications:
- Image generation.  
- Deepfakes.  
- Data augmentation.  

---

## Q18. What are Transformers?

**Answer:**  
Transformers are architectures based on the **self-attention mechanism**, which allows the model to focus on different parts of input sequences.  

Advantages over RNNs:
- Parallel processing (not sequential).  
- Handle very long dependencies.  

Key components:
- **Encoder-Decoder structure.**  
- **Multi-Head Self-Attention.**  
- **Positional Encoding.**  

Used in NLP models like BERT, GPT, T5.

---

## Q19. What is Transfer Learning?

**Answer:**  
Transfer learning is the process of using a **pre-trained model** (trained on a large dataset like ImageNet) and fine-tuning it on a new, smaller dataset.  

Advantages:
- Saves training time.  
- Works well with limited data.  
- Pre-trained models learn **general features** useful for multiple tasks.  

---

## Q20. What are the main challenges in Neural Networks?

**Answer:**  
1. **Overfitting:** solved by regularization, dropout.  
2. **Vanishing/Exploding Gradients:** solved by ReLU, batch norm, ResNets.  
3. **Computational Cost:** deep models require GPUs/TPUs.  
4. **Interpretability:** hard to understand how predictions are made.  
5. **Bias & Fairness:** models may learn unwanted biases from data.  
6. **Energy Consumption:** large models like GPT consume huge power.  

---
