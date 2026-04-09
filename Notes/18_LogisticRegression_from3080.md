<head>
<title>Logistic Regression</title>
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>

# Logistic Regression
Reading
* James et al., 4.3 Logistic Regression

## Classification Models
<img src="https://raw.githubusercontent.com/drolsonmi/math3480/refs/heads/main/Notes/Images/3480_ML_Landscape.png" width=550 alt="Machine Learning Landscape">

## The Sigmoid Function
Provide a situation and show a graph

$$f(x) = \frac{1}{1+e^{-x}}$$

## Logistic Regression Equation

$$z = \beta_0 + \beta_1 x$$

*log odds* function

$$\log\left(\frac{p(x)}{1-p(x)}\right) = z$$

$$\frac{p(x)}{1-p(x)} = e^z$$

$$p(x) = (1-p(x))e^z$$

$$p(x)(e^z+1) = e^z$$

$$p(x) = \frac{e^z}{e^z+1}$$

$$p(x) = \frac{1}{1+e^{-z}} = \frac{1}{1+e^{-(\beta_0 + \beta_1 x)}}$$

> Graph on Desmos

## Finding the Coefficients


### An Intro to Gradient Descent
* Error equation
* Goal is to minimize error
* What is a gradient? It's a slope
* Find gradient of error
* Take a small step towards the minimum
* After many small steps (maybe 1,000's), we reach a minimum error
* Common training method for many models

$$J(\beta) = -\frac{1}{m}\sum_{i=0}^{m-1}[y^{(i)}\log(\hat{p}^{(i)}) + (1-y^{(i)})log(1-\hat{p}^{(i)})] \qquad \hat{p} = \sigma(f(x)) = \sigma(\beta_0 + \beta_1x)$$

where $x^{(i)}$ is the $i$-th datapoint, $y^{(i)}$ is it's matching y-value. We want to minimize the cost function.

$$\frac{\partial J}{\partial \beta_0} = \frac{1}{m}\sum_{i=0}^{m-1}\left(\hat{p}^{(i)} - y^{(i)}\right) \qquad \frac{\partial J}{\partial \beta_1} = \frac{1}{m}\sum_{i=0}^{m-1}\left(\hat{p}^{(i)} - y^{(i)}\right)x^{(i)} = \frac{1}{m}\left[\left(\hat{p}-y\right)\cdot X\right]$$

The gradient descent algorithm where $\eta$ is the learning rate (other textbooks may use $a$ or $\delta$ to depict the learning rate - whatever it is, it's a small number to indicate the step size desired to converge toward the solution):

$$\beta_0' = \beta_0 - \eta_0 \frac{\partial J}{\partial \beta_0} \qquad \beta_1' = \beta_1 - \eta_1 \frac{\partial J}{\partial \beta_1}$$

In practice, we generally just continue until $\tfrac{\partial J}{\partial \beta_0}$ and $\tfrac{\partial J}{\partial \beta_1}$ fall below a certain threshold, indicating that we are close to our solution.

## Multiple Logistic Regression
$$p(x) = \frac{1}{1+e^{-z}} \qquad z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$