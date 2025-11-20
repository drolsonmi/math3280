# Logistic Regression
Reading
* Geron, Chapter 4 (pages 142-151)

Gradient Descent can also be used in Logistic Regression.

Linear Regression is a *regression* model - it can be used to predict values (continuous model).

Logistic Regression is a *classification* model - it can be used to predict categories (discrete model).
* Binomial responses, such as Yes/No
* Based on probabilities

## How Logistic Regression works
Like linear regression, there is a linear equation used in the prediction.
$$y = \theta_0 + \theta_1x + \dots = \vec{\theta}\cdot \vec{x}$$

Unlike linear regression, however, a function is applied to this linear equation. Specifically, we apply the linear equation to a sigmoid function:
$$\sigma(u) = \frac{1}{1+e^{-u}}$$

> Plot on Desmos

Our predictions are calculated as a probability based on the sigmoid function:
$$p' = \sigma (\vec{\theta} \cdot \vec{x}) = \frac{1}{1+e^{-\theta_0-\theta_1x}}$$

> Plot on Desmos and vary $\theta_0$ and $\theta_1$

* If $\theta_0$ varies, the position of the transition changes
* If $\theta_1$ varies, the steepness of the transition changes
    * Sigmoid reverses direction if $\theta_1<0$


What do we use for $\theta_0$ and $\theta_1$?
* Random numbers
* Use Gradient Descent to fine-tune

For Gradient Descent, we need a loss function

## The Log-Loss Function
Our Loss Function (sometimes called a Cost Function) is based on this piecewise function for an individual prediction:
$$c(\vec{\theta}) = \begin{cases}-\log(p') & \text{if }y=1 \\ -log(1-p') & \text{if }y = 0\end{cases}$$

> Plot on Desmos
```python
f(x) = t0 + t1*x
p(x) = 1 / (1 + exp(-f(x)))
y = -log(p(x))
y = -log(1-p(x))
```

* If $y=1$, then $-log(p')$ drops to 0 as x increases
* If $y=0$, then $-log(1-p')$ drops to 0 as x decreases

The Loss function itself combines these relationships:
$$J(\vec{\theta}) = -\frac{1}{m}\sum \left[y_i \log(p'_i) + (1-y_i) \log(1-p'_i)\right]$$

This is the function we want to minimize. So, we update our model using Gradient Descent:
$$\vec{\theta}_{i+1} = \vec{\theta}_i - \eta\nabla J(\vec{\theta}_i)$$

Working out the partial derivatives:
$$\frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_i \left(\sigma(\vec{\theta}\cdot\vec{x}_i) - y_i\right) \qquad \frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_i \left(\sigma(\vec{\theta}\cdot\vec{x}_i) - y_i\right)x_i$$

> Examples in ./Code/18_LogisticRegression.ipynb