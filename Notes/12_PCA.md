<head>
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

# Principal Component Analysis

## Materials
* Brunton, Chapter 1, Section 1.5
* Geron, Chapter 8

Helpful videos:
* Serrano Academy: https://www.youtube.com/watch?v=g-Hb26agBFg
* Steve Brunton: https://www.youtube.com/watch?v=fkf4IBRSeEc&list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv


## What is the PCA?
Desmos Graphics:
* https://www.desmos.com/calculator/ywbgtwwvsi


## How to calculate the PCA
PCA is based on the SVD, and is the bedrock dimensionality reduction technique, commonly used in statistics, probability, and other data science field.
* Useful to uncover lower dimension variables from big data for analysis and to establish models

Around since 1901. Can be considered as the Statistical Interpretation of SVD
* Data-driven hierarchical coordinate system based on data to represent statistical variation in data

$$X = \begin{bmatrix}
\dots & x_1 & \dots \\
\dots & x_2 & \dots \\
 & \vdots &
\end{bmatrix}$$

Each row represents the measurements from a single experiment (consistent with statistics).

__Procedure__:
1. Compute mean row

$$\bar{x}=\frac{1}{n}\sum_{j=1}^n x_j$$

$$\bar{X}=\begin{bmatrix}1\\1\\\vdots\\1\end{bmatrix}\begin{bmatrix}\qquad\bar{x}\qquad\end{bmatrix}$$

  * Consider $\bar{X}$ to be a matrix of nothing but the average value for each column

2. Subtract the mean

$$B=X-\bar{X}$$

3. Covariance matrix (correlation matrix from SVD) of rows of B:

$$C=\frac{1}{n-1}B^TB$$

The covariance matrix will look like this:
$$C = \begin{bmatrix}
  var(x_1)     & covar(x_1,x_2) & covar(x_1,x_3) & \dots \\
covar(x_2,x_1) &   var(x_2)     & covar(x_2,x_3) & \dots \\
covar(x_3,x_1) & covar(x_3,x_2) &   var(x_3)     & \dots \\
\vdots         &   \vdots       &  \vdots        & \ddots  
\end{bmatrix}$$

4. Compute eigenvalues and eigenvectors of covariance matrix $C$

$$Cv=\lambda v \qquad\to\qquad v^TCv = \lambda v^Tv \qquad\to\qquad v^T B^T B v = \lambda$$

Do this for all possible $v$ and find their eigenvalues. Then, we have a set of eigenvalues we'll put in a matrix $V$ and the corresponding eigenvalues, which we'll put in a matrix $D$

$$CV=VD$$

Make a new matrix, $T=BV$

* $T$ is a matrix of principal components
* $V$ are the loadings

This is using traditional PCA notation. But we can relate this back to SVDs:

$$B=U\Sigma V^T \qquad T=BV=U\Sigma V^T V = U\Sigma$$

We can thus use the SVD to get the principal components and loadings, which are results of the very important statistical calculations.

The eigenvalues of the matrix $D$ would be the square of the singular values $\lambda = \sigma^2$, which represents the variance of its corresponding principal component from the data.

The percentage of the variance captured in the first $r$ principal components would be,

$$\frac{\sum_{k=1}^r \lambda_k}{\sum_{k=1}^n \lambda_k}$$

We can use this select the first $r$ components to capture a given amount of variance (say, 95%).

> [12 Principal Component Examples](./Code/12_PCA.ipynb)
> * Small example
> * Larger example
> * With Linear Regression on Diabetes dataset