<head>
<title>Distance Measures</title>
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

# Distance Measures

## Linear Algebra Review

### Vectors
A vector is a mathematical tool that expresses both magnitude and direction.
$$\vec{y} = \begin{bmatrix} 3 \\ 2 \end{bmatrix} \qquad \vec{z} = \begin{bmatrix} 1 \\ 4 \end{bmatrix}$$

### Vector Norms
We learned about norms in our [MATH 3080](https://github.com/drolsonmi/math3080) class. Below is a review of norms.

A norm is a way to calculate the length of a vector. Another way to say it is to measure the distance from the tail to the tip. But how do we measure the distance?

Most common method is known at the Pythagorean Theorem,
$$\lVert x \rVert = \sqrt{x_0^2 + x_1^2 + x_2^2 + \dots}$$

This is actually one type of norm. We can create a simple equation to describe the norm:
$$\lVert x \rVert_2 = \sqrt{\sum_i x_i^2}$$

But there are other norms which measure the distance of a vector. We can generalize this equation for any norm:
$$\lVert x \rVert_n = \sqrt[n]{\sum_i |x_i|^n}\tag{$L_n$ norm}$$

The most common norms are:
1. $L_1$ norm (also known as the Manhattan Distance). This essentially finds the distance by following each component separately, as if you are walking from point A to point B on the streets of Manhattan.
$$\lVert x \rVert_1 = \sqrt[1]{\sum_i |x_i|^1} = \sum_i |x_i| = |x_0| + |x_1| + |x_2| + \dots\tag{$L_1$ norm}$$
$$\lVert y \rVert_1 = 3 + 2 = 5$$

2. $L_2$ norm (also known as both the Pythagorean Theorem and the Euclidean Distance).
$$\lVert x \rVert_2 = \sqrt[2]{\sum_i |x_i|^2} = \sqrt{x_0^2+x_1^2+x_2^2+\dots}\tag{$L_2$ norm}$$
$$\lVert y \rVert_2 = \sqrt{3^2 + 2^2} = \sqrt{9+4} = \sqrt{13} = 3.606$$

3. $L_\infty$ norm. With the infinity-norm, all components are raised to an extremely large power. Small components become insignificant to the largest component, so we end up with just the largest component, or the maximum.
$$\lVert x \rVert_\infty = \sqrt[\infty]{\sum_i |x_i|^\infty} = \sqrt[\infty]{max_i |x_i|^\infty} = \max_i |x_i|\tag{$L_\infty$ norm}$$
$$\lVert y \rVert_\infty = \max\{3,2\} = 3$$

Other norms are a measure of distance between the $L_2$ and $L_\infty$ norms. For example, the $L_3$ norm would measure a distance, but the largest components are weighed a little more than the smaller components.
$$\lVert x \rVert_3 = \sqrt[3]{\sum_i |x_i|^3} = \sqrt[3]{x_0^3 + x_1^3 + x_2^3 + \dots}\tag{$L_3$ norm}$$
$$\lVert y \rVert_3 = \sqrt[3]{3^3 + 2^3} = \sqrt[3]{27+8} = \sqrt[3]{36} = 3.302$$

The $L_{12}$ norm would be similar, but the weight of the largest components are weighed even heavier toward the distance than the $L_3$ norm.
$$\lVert x \rVert_{12} = \sqrt[12]{\sum_i x_i^{12}} = \sqrt[12]{x_0^{12} + x_1^{12} + x_2^{12} + \dots}\tag{$L_{12}$ norm}$$
$$\lVert y \rVert_{12} = \sqrt[12]{3^{12} + 2^{12}} = \sqrt[12]{531441+4096} = \sqrt[3]{535537} = 3.0019$$

Generally, if a norm is given without a number, we default to the Euclidean norm (L2 norm).
$$||\vec{v}|| = ||\vec{v}||_2$$

Conceptually, the $L_1$ and $L_2$ norms are easy. Even the $L_\infty$ norm is simple enough to understand. But why would we want other norms, like the $L_3$ or $L_{12}$ norms?
* Physical definition of Distance: Amount of space between two points
* Reworded definition of Distance: Measure of how close two points are
* Applied to Data Science: Measure of how close two measurements are



### Basis Vectors
A basis vector is a vector of length 1 in the direction of a vector. The three most common basis vectors are the unit vectors in the x, y, and z directions:
$$\hat{i} = \begin{bmatrix}1 \\ 0 \\ 0\end{bmatrix} \qquad \hat{j} = \begin{bmatrix}0 \\ 1 \\ 0\end{bmatrix} \qquad \hat{z} = \begin{bmatrix}0 \\ 0 \\ 1\end{bmatrix}$$

We can easily find a basis vector by dividing out the magnitude from any vector:
$$\hat{v} = \frac{\vec{v}}{||\vec{v}||_2}$$

### Dot Product (inner product)
We can multiply two vectors together many different ways. The most common method is the dot product. Just multiply the corresponding components for the two vectors together then add the results up.
$$\vec{v}\cdot\vec{w} = \sum_i v_iw_i$$
$$\vec{y}\cdot\vec{z} = 3*1 + 2*4 = 3 + 8 = 11$$

Here is a more in-depth way to look at the dot product calculation:
$$\vec{v}\cdot\vec{w} = \vec{v}^T\vec{w}$$
$$\vec{y}\cdot\vec{z} = \vec{y}^T\vec{z} = \begin{bmatrix} 3 & 2\end{bmatrix}\begin{bmatrix}1 \\ 4\end{bmatrix} = 3*1+2*4=3+8=11$$

These formats are interchangeable. We'll see both forms throughout the semester.

The dot product of a vector with itself is related to the Euclidean Norm of a vector:
$$\vec{v}\cdot\vec{v} = \sum v_i^2 = ||\vec{v}||^2_2 \qquad \to \qquad ||\vec{v}||_2 = \sqrt{\vec{v}\cdot\vec{v}}$$

A third way to calaculate the dot product is using norms:
$$\vec{v}\cdot\vec{w} = ||v||_2||w||_2\cos\theta$$

But what is the dot product? The dot product is often what we call a *projection*. It describes what component of vector $\vec{v}$ is pointing in the same direction of vector $\vec{w}$.
> Examples:
> * Project vector $\vec{z}$ onto the x-axis and onto the y-axis
> * Project vector $\vec{z}$ onto vector $\vec{y}$
> 
> Consider demonstrating on Desmos

* The result of the dot product is equal to (the component of vector $\vec{v}$ in the direction of $\vec{x}$) times (the magnitude of $\vec{w})$

### Outer product
An outer product will multiply to vectors together in a way that the result is a matrix showing the product of all pairs of coordinates:

$$\vec{v}\otimes\vec{w} = \vec{v}\vec{w}^T$$
$$\vec{y}\otimes\vec{z} = \vec{y}\vec{z}^T = \begin{bmatrix} 3 \\ 2\end{bmatrix}\begin{bmatrix}1 & 4\end{bmatrix} = \begin{bmatrix} 3*1 & 3*4 \\ 2*1 & 2*4 \end{bmatrix} = \begin{bmatrix} 3 & 12 \\ 2 & 8\end{bmatrix}$$

### Cosine Distance
A low physical distance implies that two points are almost identical. However, is there a way to measure that two points have similar attributes despite being very different? 

Instead of looking at the physical distance between two points, we look at whether two points are pointing in similar, opposite, or perpendicular directions. We do this using the cosine-definition of the dot product.
$$\vec{x}\cdot\vec{y} = \lVert x \rVert\lVert y \rVert \cos\theta \qquad\qquad \cos\theta = \frac{\vec{x}\cdot\vec{y}}{\lVert x \rVert\lVert y \rVert} \qquad\qquad \theta = \arccos\left(\frac{\vec{x}\cdot\vec{y}}{\lVert x \rVert\lVert y \rVert}\right)$$

__The angle $\theta$ is the cosine distance.__ However, we often look at $\cos\theta$ instead as it becomes more intuitive.

| The two points are             | $\theta$                     | $\cos\theta$              |
| -----------------------------: | :--------------------------: | :-----------------------: |
|                                | $0 \le \theta \le 2\pi$      | $-1 \le \cos\theta \le 1$ |
|       in similar directions if | $\theta\approx 0$ or $2\pi$  | $\cos\theta \approx 1$    |
| in perpendicular directions if | $\theta\approx\frac{\pi}{2}$ | $\cos\theta \approx 0$    |
|     in oppposite directions if | $\theta\approx\pi$           | $\cos\theta \approx -1$   |


### Vector Transformations (using Matrices)
We can write a series of vectors together as a matrix:

$$A = \begin{bmatrix} a_{00} & a_{01} & \dots & a_{0n} \\ a_{10} & a_{11} & \dots & a_{1n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m0} & a_{m1} & \dots & a_{mn}\end{bmatrix}$$
$$B = \begin{bmatrix} 10 & 6 \\ 8 & 4\end{bmatrix}$$

Then, we can multiply the matrix by a vector (dot product of each row with the vector):

$$B\vec{y} = \begin{bmatrix} 10 & 6 \\ 8 & 4\end{bmatrix}\begin{bmatrix} 3 \\ 2\end{bmatrix} = \begin{bmatrix} 10*3 + 6*2 \\ 8*3 + 4*2\end{bmatrix} = \begin{bmatrix} 42 \\ 32\end{bmatrix}$$

This can be useful to transform a vector to a new coordinate system. Our most traditional 2D coordinate system (known as the Cartesian Coordinate System) can be written as a matrix of two vectors: one for the x-axis and one for the y-axis.

$$\begin{bmatrix} 1 & 0 \\ 0 & 1\end{bmatrix}$$

Each column would be a vector, and the combination of the vectors are known as __basis vectors__ which describe the coordinate system. Using this, we can create new coordinate systems. That is, we can measure a vector against any other set of vectors we want. Then when we multiply a matrix by our vector, it transforms our vector onto the new coordinate system.

For example, we can rotate the coordinate system 90 degrees counter-clockwise:

$$R = \begin{bmatrix} 0 & -1 \\ 1 & 0\end{bmatrix}$$
$$R\vec{y} = \begin{bmatrix} 0 & -1 \\ 1 & 0\end{bmatrix}\begin{bmatrix} 3 \\ 2\end{bmatrix} = \begin{bmatrix} 0*3+(-1)*2 \\ 1*3+0*2\end{bmatrix} = \begin{bmatrix} -2 \\ 3\end{bmatrix}$$

Or we can use any other two vectors we want to create a matrix transformation:

$$K = \begin{bmatrix} 3 & -1 \\ 4 & -2\end{bmatrix}$$
$$K\vec{y} = \begin{bmatrix} 3 & -1 \\ 4 & -2\end{bmatrix}\begin{bmatrix} 3 \\ 2\end{bmatrix} = \begin{bmatrix} 3*3+(-1)*2 \\ 4*3+(-2)*2\end{bmatrix} = \begin{bmatrix} 7 \\ 8\end{bmatrix}$$

(Draw each transformation by drawing the vector on the Cartesian Coordinates, then beside it draw the vector relative to the new basis vectors.)

This will become a very useful tool in Data Science. For example, we can find the line of best fit in linear regression with variance, covariance, and correlation. But this is more difficult in multi-linear regression. So, we use matrix transformations to simplify the process.

### Eigenvectors and Eigenvalues
(If there is time, we will address this. If not, then we can address this in a later lecture.)

With some transformations, the direction of our vector doesn't change, but its length might. Such vectors are known as __eigenvectors__.
$$A\vec{x} = \lambda\vec{x}$$

$\lambda$ is a multiplier known as an __eigenvalue__. 

$$K\vec{z} = \begin{bmatrix} 3 & -1 \\ 4 & -2\end{bmatrix}\begin{bmatrix} 1 \\ 4\end{bmatrix} = \begin{bmatrix} 3*1+(-1)*4 \\ 4*1+(-2)*4\end{bmatrix} = \begin{bmatrix} -1 \\ -4\end{bmatrix} = (-1)\begin{bmatrix} 1 \\ 4\end{bmatrix} = (-1)\vec{z}$$

So, $z = [1,4]$ is an eigenvector of $K$ with an eigenvalue of $-1$. (Draw the transformation.)

Does $K$ have other eigenvalue/eigenvector pairs?
$$\begin{align*}K\vec{x} &= \lambda\vec{x} \\
  \begin{bmatrix} 3 & -1 \\ 4 & -2\end{bmatrix}\begin{bmatrix} x_0 \\ x_1\end{bmatrix} &= \lambda \begin{bmatrix} x_0 \\ x_1\end{bmatrix}\\ 
  \begin{bmatrix}3x_0-x_1 \\ 4x_0-2x_1\end{bmatrix} &= \begin{bmatrix} \lambda x_0 \\ \lambda x_1\end{bmatrix} \\
  3x_0-x_1 = \lambda x_0 \qquad&\qquad 4x_0-2x_1=\lambda x_1 \\
  (3-\lambda)x_0 - x_1 = 0 \qquad&\qquad 4x_0-(2+\lambda)x_1 = 0\\
  x_1 = (3-\lambda)x_0 \qquad & \qquad 4x_0-(2+\lambda)(3-\lambda)x_0 = 0 \\
  &\qquad 4 - (6+\lambda-\lambda^2) = 0 \\
  &\qquad \lambda^2 - \lambda - 2 = 0 \\
  &\qquad \lambda = \frac{1\pm\sqrt{1-4(1)(-2)}}{2(1)} \\
  &\qquad \lambda = \{2,-1\}
\end{align*}$$

So, we have two eigenvalues: $\lambda = 2$ and $\lambda = -1$. We already saw the eigenvalue $\lambda=-1$ has an eigenvector of $\vec{x}=\begin{bmatrix}1\\4\end{bmatrix}$. What's the eigenvector for $\lambda = 2$?
$$\begin{align*}
  3x_0-x_1 = 2x_0 \qquad&\qquad 4x_0-2x_1=2x_1 \\
  x_0-x_1 = 0 \qquad&\qquad 4x_0-4x_1=0 \\
  x_0 = x_1 \qquad&\qquad x_0=x_1
\end{align*}$$
All we know is that $x_0=x_1$, which means,
$$\vec{x} = \begin{bmatrix}x_0\\x_1\end{bmatrix} = \begin{bmatrix}x_0\\x_0\end{bmatrix} = x_0\begin{bmatrix}1\\1\end{bmatrix}$$
The eigenvector for $\lambda=2$ is $\vec{x}=\begin{bmatrix}1\\1\end{bmatrix}$.

