<head>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>

# 14: Linear Regression

## Recall from Linear Regression
If we have 2 variables that have a linear relationship, we can find a line that approximates the behavior of the data.
* If there are only 2 points, we can get a perfect line between the two points
* If there are 3 points, then if we're lucky, we can get a perfect line between all 3 points
    * More likely, it won't be possible. So what do we do?
    * We find the line that minimizes the error

> * [Desmos: Linear Regression](https://www.desmos.com/calculator/ezcqkdityj)

In the 2-variable case, the equation will be of the form,
$$y=\theta_0+\theta_1x$$

where
* $x$ is the input variable (independent variable)
* $y$ is the output variable (dependent variable)
* $\theta_0$ is the bias (commonly known as the y-intercept)
* $\theta_1$ is the parameter (commonly known as the slope)

How do we find $\theta_0$ and $\theta_1$?
* Find the central point $(\bar{x},\bar{y})$
* Find the slope 
$$\theta_1=\frac{s_y}{s_x}r$$
where $r$ is the correlation coefficient, found as,
    $$r=\frac{1}{n-1}\sum_i \frac{x_i-\bar{x}}{s_x}\frac{y_i-\bar{y}}{s_y}$$
* The equation is,
$$y-\bar{y}=\theta_1(x-\bar{x}) \qquad\to\qquad y=\theta_1x + (\bar{y} - \theta_1\bar{x})$$
* The bias is the last term
$$\theta_0=\bar{y}-\theta_1\bar{x} \qquad\to\qquad y=\theta_0+\theta_1x$$

> * [14_LinearRegression.ipynb - 2D Example - Using traditional calculations](./Code/14_LinearRegression.ipynb)

## Using Linear Algebra
$$\theta_0 + x\theta_1=y$$

We can rearrange this as a linear system of equations:
$$\begin{bmatrix}1 & x\end{bmatrix}\begin{bmatrix}\theta_0 \\ \theta_1\end{bmatrix} = \begin{bmatrix}y\end{bmatrix}$$

If we have $x$ and $y$, we will want to find the right $\theta_0$ and $\theta_1$. However, to do this, we'll need 2 observations to make a line.

$$\begin{bmatrix}1 & x_0 \\ 1 & x_1\end{bmatrix}\begin{bmatrix}\theta_0 \\ \theta_1\end{bmatrix} = \begin{bmatrix}y_0 \\ y_1\end{bmatrix} \qquad\qquad X\Theta=Y$$

Assuming $X$ is invertible, we can find $\Theta$ by taking $X^{-1}$.
$$\Theta=X^{-1}Y = \begin{bmatrix}1 & x_0 \\ 1 & x_1\end{bmatrix}^{-1}\begin{bmatrix}y_0 \\ y_1\end{bmatrix}$$

There are a few problems:
1. $X$ isn't always invertible
2. Even if it is invertible, it's based on only two points and will not be accurate for multiple points
3. If $X$ isn't a square (which is most often the case), it's almost never invertible

## Shapes of matrices
Let's take the general case,
$$Ax=b$$

We can solve for x only if A is square
* Underdetermined matrix
    * n < m

    $$\begin{bmatrix}a_{00} & a_{01} & a_{02} & \dots & \dots & a_{0m} \\
                     a_{10} & a_{11} & a_{12} & \dots & \dots & a_{1m} \\
                     \vdots & \vdots & \vdots &       &       & \vdots \\
                     a_{n0} & a_{n1} & a_{n2} & \dots & \dots & a_{nm} \end{bmatrix}\begin{bmatrix}x_0 \\ x_1 \\ x_2 \\ \vdots \\ \vdots \\ x_m\end{bmatrix} = \begin{bmatrix}b_0 \\ b_1 \\ \vdots \\ b_n\end{bmatrix}$$
    * Often has infinitely many solutions (can come up with a few exceptions)

* Overdetermined matrix
    * n > m

    $$\begin{bmatrix}a_{00} & a_{01} & \dots & a_{0m} \\
                     a_{10} & a_{11} & \dots & a_{1m} \\
                     a_{20} & a_{21} & \dots & a_{2m} \\
                     \vdots & \vdots &       & \vdots \\
                     \vdots & \vdots &       & \vdots \\
                     a_{n0} & a_{n1} & \dots & a_{nm} \end{bmatrix}\begin{bmatrix}x_0 \\ x_1 \\ \vdots \\ x_m\end{bmatrix} = \begin{bmatrix}b_0 \\ b_1 \\ b_2 \\ \vdots \\ \vdots \\ b_n\end{bmatrix}$$
    * Often has zero solutions (can come up with a few exceptions)

### Details on number of solutions
How many solutions $x$ are there in $Ax=b$? First, a few definitions of some subspaces: 
* $col(A)$ is the column space, or the linear combination of the vectors in the matrix $A$
    * Note: This is very similar to $span(A)$, but $span()$ is a linear combination of a set of vectors. 
        * $col(A)\subset span(A)$
    * $col(A) = col(\hat{U})$
* $ker(A^T)$ is the orthogonal complement of $col(A)$
* $row(A)$ is the row space, or the set of all vectors that can be spanned by the rows of $A$
    * $row(A) = row(V^T) = col(V)$
* $ker(A)$ is the null space, or the set of all vectors s.t. $Ax=0$

Let's see a specific example. Let $A$ be a matrix with basis vectors $\vec{a}_i$:
$$A = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix} \qquad\to\qquad \vec{a}_1 = \begin{bmatrix}1 \\ 4 \\ 7\end{bmatrix} \qquad \vec{a}_2 = \begin{bmatrix}2 \\ 5 \\ 8\end{bmatrix} \qquad \vec{a}_3 = \begin{bmatrix}3 \\ 6 \\ 9\end{bmatrix}$$

If we multiply $A$ by $x$, we'll get a linear combination of the $\vec{a}_i$ vectors.
$$Ax = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}\begin{bmatrix}10 \\ 20 \\ 30\end{bmatrix}= 10\begin{bmatrix}1 \\ 4 \\ 7\end{bmatrix} + 20\begin{bmatrix}2 \\ 5 \\ 8\end{bmatrix} + 30 \begin{bmatrix}3 \\ 6 \\ 9\end{bmatrix} = \begin{bmatrix}110 \\ 320 \\ 500\end{bmatrix} = b$$

$b$ is a linear combination of the vectors of $A$, so $b$ is in the column space of $A$ (a possible linear combination of $A$). This is what is meant by $col(A)$.

Back to how many solutions there are:
1. There is only 1 solution iff $b\in col(A)$ and $dim(ker(A)) = 0$
2. There are infinitely many solutions if $b\in col(A)$ and $dim(ker(A)) \ne 0$
    * $dim(ker(A))\ne 0$ means there is at least one solution to $A$ s.t. $Ax=0$
        * Let $x_{null}$ satisfy $Ax_{null} = 0$ (that is, $x_{null} \in ker(A)$)
        * Let $x$ be a solution to $Ax=b$
        * Then $A(x+x_{null}) = b+0 = b$ is also a solution
    * There are an infinite number of linear combinations of $x$ and $x_{null}$, so there are infinitely many solutions
3. There are no solutions if $b \notin col(A)$

## Using SVDs to find an approximate inverse
Let's use SVDs
$$A=\hat{U}\hat{\Sigma}V^T$$
$$Ax=b$$
$$\hat{U}\hat{\Sigma}V^Tx=b$$
$$V\Sigma^{-1}U^T U\Sigma V^T x = V\Sigma^{-1}U^T b$$
$$x = V\Sigma^{-1}U^T b$$
$$x = A^\dagger b$$


where $A^\dagger$ is known as the __Moore-Penrose__ (left) __Pseudo Inverse__. We can use $A^\dagger$ to approximate $x$
$$x\approx \tilde{x} = A^\dagger b$$

In the underdetermined case, there are an infinite number of solutions. Which one is right? We generally say that the *minimum-norm solution* is the one where $\min{||\tilde{x}||_2}$ such that $A\tilde{x}=b$.

In the overdetermined solution, we can find the $\tilde{x}$ that minimizes the error:
$$\min||A\tilde{x}-b||_2$$
This is known as the least squares solution.

So, how well does this work? To find out, let's plug everything back in:
$$A=\hat{U}\hat{\Sigma}V^T \qquad \tilde{x} = A^\dagger b = V\Sigma^{-1}U^T b$$
$$A\tilde{x} = \hat{U}\Sigma V^T V\Sigma^{-1} \hat{U}^T b$$
$$A\tilde{x} = \hat{U}\hat{U}^T b$$

Remember that $\hat{U}$ is not unitary, so $\hat{U}\hat{U}^T\ne \mathbb{I}$. So what is this? $\hat{U}\hat{U}^T b$ is the projection of $b$ onto $span(\hat{U}) = span(A)$. 
* $span(A)$ is the set of all possible linear combinations of the vectors in $A$

In the underdetermined case, this makes sense because the only way to get a solution of $Ax=b$ is if $b$ is in the column space (or the span) of $A$. If $b$ doesn't appear as a column of $A$, there are plenty of columns to create a linear combination of the different columns to find $b$.

In the overdetermined case, the only way to get a solution is if $b$ is a scaled columnn of $A$. Since there are more rows than column, it is much more likely that there is one component of $b$ that can't be found using a linear combination of the columns of $A$. So, the only way to guarantee a solution is if $b$ is a column of $A$. But by using the SVD, we get $\hat{U}\hat{U}^Tb$ which is a project of $b$ onto the span of $\hat{U}$ (or of $A$), which will determine a solution.

## Linear Regression usind SVDs
### 2D case
Let's go back to the 2D case. We want multiple observations to make the best line.
$$\begin{bmatrix}1 & x_0 \\ 1 & x_1 \\ 1 & x_2 \\ 1 & x_3 \\ 1 & x_4 \\ 1 & x_5 \end{bmatrix}\begin{bmatrix}\theta_0 \\ \theta_1\end{bmatrix} = \begin{bmatrix}y_0 \\ y_1 \\ y_2 \\ y_3 \\ y_4 \\ y_5\end{bmatrix}$$

If we only have 2 x values, then we can take the inverse of $X$ to solve for $\Theta$. However, this only gives a line between the two points. We want a line that fits through all points. This means we need a lot of data, and we can't take an inverse of $X$. What do we do? We use the Moore-Penrose Pseudo Inverse.

> [14_LinearRegression.ipynb - 2D Example - Using the SVD method](./Code/14_LinearRegression.ipynb)

### Multilinear Regression
Let's add a second variable. Now $\Theta$ has one bias ($\theta_0$) and two parameters ($\theta_1$ and $\theta_2$).
$$\begin{bmatrix}1 & x_{01} & x_{02} \\ 1 & x_{11} & x_{12} \\ 1 & x_{21} & x_{22} \\ 1 & x_{31} & x_{32} \\ 1 & x_{41} & x_{42} \\ 1 & x_{51} & x_{52} \end{bmatrix}\begin{bmatrix}\theta_0 \\ \theta_1 \\ \theta_2\end{bmatrix} = \begin{bmatrix}y_1 \\ y_2 \\ y_3 \\ y_4 \\ y_5\end{bmatrix}$$



We can adapt this to as many dimensions as we want.
$$\begin{bmatrix}1 & x_{01} & x_{02} & \dots & x_{0m} \\ 1 & x_{11} & x_{12} & \dots & x_{1m} \\ 1 & x_{21} & x_{22} & \dots & x_{2m} \\ 1 & x_{31} & x_{32} & \dots & x_{3m} \\ 1 & x_{41} & x_{42} & \dots & x_{4m} \\ 1 & x_{51} & x_{52} & \dots & x_{5m} \\ \vdots & \vdots & \vdots & & \vdots \\  1 & x_{n1} & x_{n2} & \dots & x_{nm}\end{bmatrix}\begin{bmatrix}\theta_0 \\ \theta_1 \\ \theta_2 \\ \vdots \\ \theta_m \end{bmatrix} = \begin{bmatrix}y_1 \\ y_2 \\ y_3 \\ y_4 \\ y_5 \\ \vdots \\ y_n\end{bmatrix}$$

For example,
* Row $i$ indicates a heart patient
* $y_i$ is the likelhood of developing cardiac problems
* $x_{i1}$ might be your age
* $x_{i2}$ might be your weight
* $x_{i3}$ might be a measure of your diet
* ...

The goal is to use our $n$ people to determine the best $\Theta$ so that when we use it to make a linear combination of a new patient's measurements $X$, we can make a prediction $y$.

> [14_LinearRegression.ipynb - Mutlilinear Regression](./Code/14_LinearRegression.ipynb)