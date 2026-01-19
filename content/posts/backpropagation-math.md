+++
title = "backpropagation for dorks"
date = "2026-01-09T22:55:03Z"
author = "josh"
tags = ["ml", "math"]
description = "the boring bit"
showFullContent = false
readingTime = true
+++

this is a companion post to [neural nets from scratch in rust](/posts/first-post/), covering the mathematical foundations of backpropagation.

# the setup

## $ y = f(WX + b) $

the holy grail formula that underpins neural networks is just a linear transform that you then apply a differentiable nonlinear transformation to.

lets consider a single layer neural network and define our shapes:

- $X \in \mathbb{R}^{N \times P}$ is our input data, where $N$ is the number of samples and $P$ is the feature space dimension
- $W \in \mathbb{R}^{P \times C}$ is our weight matrix, where $C$ is the number of output classes
- $b \in \mathbb{R}^{1 \times C}$ is our bias vector
- $y \in \mathbb{R}^{N \times C}$ is our output logits

the forward pass computes:

$$z = XW + b$$

where $z \in \mathbb{R}^{N \times C}$ are the raw **logits** before the activation function.

**what are logits?** logits are the unnormalized scores for each class - they're just the raw output of the linear transformation before we turn them into probabilities. they can be any real number (positive, negative, large, small) and don't need to sum to 1 or be bounded. think of them as "confidence scores" that the network assigns to each class, but they're not yet interpretable as probabilities.

we need logits as an intermediate step because:
1. the linear transformation $XW + b$ produces unbounded real values
2. we want probabilities (bounded between 0 and 1, summing to 1) for classification
3. softmax bridges this gap by converting logits to probabilities

for a classification task, we'll use the softmax activation to turn these logits into probabilities:

$$y_i = \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

this gives us a probability distribution over classes for each sample.

## the loss function

for multi-class classification, we use the cross-entropy loss. given true labels $t \in \{0, 1, ..., C-1\}^N$ (one label per sample), we first convert them to one-hot vectors $T \in \mathbb{R}^{N \times C}$, where $T_{ij} = 1$ if sample $i$ has label $j$, and $0$ otherwise.

the cross-entropy loss is:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{C} T_{ij} \log(y_{ij})$$

or more compactly:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log(y_{i,t_i})$$

where $t_i$ is the true class label for sample $i$.

**why cross-entropy?** cross-entropy comes from information theory and measures the difference between two probability distributions. if $T$ is the true distribution (one-hot encoded labels) and $y$ is our predicted distribution, cross-entropy measures how many bits of information we need on average to encode the true distribution using our predicted distribution. 

when our predictions match the true labels perfectly, cross-entropy equals the entropy of the true distribution. when our predictions are wrong, cross-entropy is larger - the difference is called the KL divergence. minimizing cross-entropy is equivalent to minimizing KL divergence between predicted and true distributions, which is exactly what we want.

## backpropagation

now for the fun part - computing gradients. we need $\frac{\partial L}{\partial W}$ and $\frac{\partial L}{\partial b}$ to update our parameters.

using the chain rule:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial W}$$

the gradient of cross-entropy loss with respect to the logits after softmax has a beautiful closed form:

$$\frac{\partial L}{\partial z} = \frac{1}{N}(y - T)$$

where $y$ are our softmax predictions and $T$ are the one-hot encoded targets. this is why softmax + cross-entropy is so popular - the gradient simplifies elegantly.

now we need $\frac{\partial z}{\partial W}$. recall that $z = XW + b$, so:

$$\frac{\partial z}{\partial W} = X^T$$

putting it together:

$$\frac{\partial L}{\partial W} = \frac{1}{N}X^T(y - T)$$

similarly for the bias:

$$\frac{\partial L}{\partial b} = \frac{1}{N}\sum_{i=1}^{N}(y_i - T_i) = \frac{1}{N}\mathbf{1}^T(y - T)$$

where $\mathbf{1}$ is a vector of ones with length $N$.

## gradient descent

with these gradients, we can update our parameters:

$$W \leftarrow W - \eta \frac{\partial L}{\partial W}$$
$$b \leftarrow b - \eta \frac{\partial L}{\partial b}$$

where $\eta$ is the learning rate, a hyperparameter that controls how big our steps are.

repeat this process enough times (forward pass, compute loss, backward pass, update weights) and your network learns to classify!
