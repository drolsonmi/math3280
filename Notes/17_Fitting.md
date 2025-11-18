# 17 Fitting
__Math 3280 - Data Mining__ : Snow College : Dr. Michael E. Olson

Reading
* Brunton, 4.1 Classical Curve Fitting
* Brunton, 4.2 Nonlinear Regression and Gradient Descent
* Geron, Chapter 4, Gradient Descent

## The Machine Learning Process
1. Obtain the data
2. Preprocessing (Cleaning and Wrangling the data)
3. Exploratory Data Analysis (EDA) (Visualization and Analyzation)
4. Model the data
5. Evaluate the model

Notes on this:
* We learned about steps 1-3 in MATH 3080
* Steps 2 and 3 are actually intertwined as Preprocessing is needed for EDA, and EDA tells us how to preprocess the data

For the remainder of the semester and into MATH 3480, we will learn about the models. However, to learn how to make good models, we need to know Evaluation techniques. This is our focus for the next week or so.


## Evaluation
### Evaluating Regression models
For evaluating regression models, we are going to define the following:
* $y$ is the true value (the labels in supervised data)
* $y'$ is the predicted value (the output of our models)

> See these evaluations in Code/17_Fitting_Evaluation.ipynb

__Mean Absolute Error (MAE)__
$$MAE=\frac{1}{n}\sum |y'-y|$$

__Sum of Square Errors (SSE)__
$$SSE = \sum (y'-y)^2$$

Why square instead of absolute value?
* The square takes any small errors and minimizes their contribution to the error
* The square takes any large errors and amplifies them (penalization)
* Models where the predications are close enough have their SSE minimized
* Models with any significant outliers are going to have those errors emphasized

However, the SSE is dependent on the sample size. A really good model with 500 datapoints could have a higher SSE than a poor model with 40 datapoints.

__Mean Square Error (MSE)__
$$MSE = \frac{1}{n}\sum(y'-y)^2$$

* Units don't always make sense

__Root Mean Square Error (RMSE) or Least-squares Error__
$$RMSE = \sqrt{\frac{1}{n}\sum(y'-y)^2}$$

__Goodness of Fit__
The MAE is taking our error to the 1st power. The RMSE is taking our error to the 2nd power. We can keep going.

$$E_k = \left(\frac{1}{n}\sum|y'-y|^k\right)^{1/k}$$

Looks a lot like our 

__Root Mean Square Log Error (RMSLE)__

$$RMSLE = \sqrt{\frac{1}{n}\sum \left[\ln(y'+1) - \ln(y+1)\right]^2}$$


### Evaluating Classification models
The key to evaluating classification models is to count the number of correct classifications. To understand these calculations, we look at the confusion matrix. Here is an example of a confusion matrix when predicting 3 categories:

|        | Predicted A | Predicted B | Predicted C |
| -----: | :---------: | :---------: | :---------: |
| True A | 54          | 4           | 9           |
| True B | 3           | 48          | 5           |
| True C | 6           | 2           | 61          |

__Accuracy__
$$accuracy = \frac{\text{\# of correct predictions}}{\text{Total \# of predictions}}$$

In our example,
$$accuracy = \frac{54+48+61}{54+3+6+4+48+2+9+5+61} = \frac{163}{192} = 0.8490$$

Not always a good measure. For example, if I have 100 images and 95 of them are of cats, then I can just guess that they are all cats and I'd have 95% accuracy. Is my model good? No. But the accuracy says it is.

For this reason, we have a few other measures. For them, we use the following definitions:
* $TP$ = True Positives (Number of events predicted and truly in a category)
* $TN$ = True Negatives (Number of events predicted and truly not in a category)
* $FP$ = False Positives (Number of events predicted in a category, but actually not)
* $FN$ = False negatives (Number of events predicted not in a category, but actually is)

__Precision__
* Of the predictions for a specific category, how many were right?
    * How many correct in a column?

$$precision = \frac{TP}{TP+FP}$$

In our example:
$$precision_A = \frac{54}{54+3+6} = \frac{54}{63} = 0.857$$
$$precision_B = \frac{48}{4+48+2} = \frac{48}{54} = 0.889$$
$$precision_C = \frac{61}{9+5+61} = \frac{61}{75} = 0.813$$

__Recall__
* For a given true category, how many did you predict correctly? 
    * How many correct in a row?

$$recall = \frac{TP}{TP+FN}$$

In our example:
$$recall_A = \frac{54}{54+4+9} = \frac{54}{67} = 0.806$$
$$recall_B = \frac{48}{3+48+5} = \frac{48}{56} = 0.857$$
$$recall_C = \frac{61}{6+2+61} = \frac{61}{69} = 0.884$$

__F1-score__
* This is the mean of the precision and recall, but we don't just add them and divide by two. Instead, we use the __harmonic mean__.
* The mean weights all entries equally
* The harmonic mean gives more weight to low values

$$f1 = \frac{2*precision*recall}{precision+recall}$$

In our Example:
$$f1_A = \frac{2*0.857*0.806}{0.857+0.806} = \frac{1.381}{1.663} = 0.830$$
$$f1_B = \frac{2*0.889*0.857}{0.889+0.857} = \frac{1.524}{1.746} = 0.873$$
$$f1_C = \frac{2*0.813*0.884}{0.813+0.884} = \frac{1.437}{1.697} = 0.847$$

## Overfitting vs Underfitting
If we increase the complexity of our model (for example, do a polynomial/nonlinear regression to higher orders), we can create models that do much better. In fact, we can get the model so that our evaluation techniques give us 0 error.

At some point, however, new data will stop benefiting from the model improvements, and those errors start to rise. Why? Extremely complex models tend to be tailored to the testing data to the point that any data outside the testing dataset is automatically considered to be an outlier.

> Code/17_Fitting_Overfitting.ipynb

## Loss Functions
When we want to optimize models, we look at a loss function (the evaulation methods above). Let's use linear and multi-linear regression as an example.
$$y' = \theta_0 + \theta_1x$$

To minimize the errors, we set up our loss function. For now, let's just consider the Least-squares error. We don't need to consider the whole equation, just the part that will actually be affected if we change our parameters.
$$E_2 = \sum(y'-y)^2$$
$$E_2 = \sum(\theta_0 + \theta_1x -y)^2$$

We want to vary $\theta_k$ to minimize the error. How do we do this? Take the derivative and set it to 0.

$$\frac{\partial E}{\partial \theta_0} = 0 \qquad \frac{\partial E}{\partial \theta_1} = 0$$

Actually performing the derivatives,

$$\frac{\partial E_2}{\partial \theta_0} = \frac{\partial}{\partial \theta_0} \sum(y'-y)^2 = \sum 2(y'-y)\frac{\partial}{\partial \theta_0}(y'-y) = \sum 2(y'-y) = \sum 2(\theta_0 + \theta_1x - y) = 0$$
$$\frac{\partial E_2}{\partial \theta_1} = \frac{\partial}{\partial \theta_1} \sum(y'-y)^2 = \sum 2(y'-y)\frac{\partial}{\partial \theta_1}(y'-y) = \sum 2(y'-y)x = \sum 2(\theta_0 + \theta_1x - y)x = 0$$

We can simplify this series of derivatives in linear algebra terms as,
$$\begin{bmatrix}n & \sum x \\ \sum x & \sum x^2\end{bmatrix}\begin{bmatrix}\theta_0 \\ \theta_1\end{bmatrix} = \begin{bmatrix}\sum y \\ \sum xy\end{bmatrix} \qquad \to \qquad X\Theta=Y$$

The solution of this system of equations will give us our model.

What about a multilinear regression? Well, it follows the same process.
$$y' = \theta_0 + \theta_1 x_1 + \theta_2 x_2$$
$$\frac{\partial E_2}{\partial \theta_0} = \frac{\partial}{\partial \theta_0} \sum(y'-y)^2 = \sum 2(y'-y)\frac{\partial}{\partial \theta_0}(y'-y) = \sum 2(y'-y) = \sum 2(\theta_0 + \theta_1 x_1 + \theta_2 x_2 - y) = 0$$
$$\frac{\partial E_2}{\partial \theta_1} = \frac{\partial}{\partial \theta_1} \sum(y'-y)^2 = \sum 2(y'-y)\frac{\partial}{\partial \theta_1}(y'-y) = \sum 2(y'-y)x_1 = \sum 2(\theta_0 + \theta_1x_1 + \theta_2x_2 - y)x_1 = 0$$
$$\frac{\partial E_2}{\partial \theta_2} = \frac{\partial}{\partial \theta_2} \sum(y'-y)^2 = \sum 2(y'-y)\frac{\partial}{\partial \theta_2}(y'-y) = \sum 2(y'-y)x_2 = \sum 2(\theta_0 + \theta_1x_1 + \theta_2x_2 - y)x_2 = 0$$

$$\begin{bmatrix}n & \sum x_1 & \sum x_2 \\ \sum x_1 & \sum x_1^2 & \sum x_1x_2 \\ \sum x_2 & \sum x_1x_2 & \sum x_2^2\end{bmatrix}\begin{bmatrix}\theta_0 \\ \theta_1 \\ \theta_2\end{bmatrix} = \begin{bmatrix}\sum y \\ \sum x_1y \\ \sum x_2y\end{bmatrix} \qquad \to \qquad X\Theta=Y$$

and so on.

However, if we consider nonlinear regression (polynomials, exponentials, etc.), these derivatives are not so straight forward. In fact, it is absolutely messy. So, what do we use instead? Gradient Descent!

## Gradient Descent
Our goal is to minimize our loss function. The problem is that the loss function depends on both $\theta_0$ and $\theta_1$. So, if I find the $\theta_0$ and $\theta_1$ that will minimize the loss function and update those values, then the loss function has changed, and it may no longer be a minimum.
* Worst case scenario, the loss function has changed so much that I am nowhere near a minimum, regardles of how close I may have been before

> * Draw two upwards parabolas to depict $\theta_0$ vs E and $\theta_1$ vs E

To avoid overshooting, don't update the parameters all the way.
1. Determine how the loss function will change with respect to each parameter $\tfrac{\partial E}{\partial \theta_k}$
2. Determine whether increase or decreasing $\theta_k$ will cause E to drop
3. Take a *small* step in the direction step 2 indicates
4. Repeat steps 1-3 until $\tfrac{\partial E}{\partial \theta_k} = 0$

To best see how this works in multiple dimensions, consider your loss function like a mountain.

If you are climbing a mountain, the steepness of the hill is measured with the gradient. The gradient is similar to a slope, but it takes more dimensions into account. If you calculate the gradient, you are finding the path of greatest ascent.

But we don't want to go uphill. We want to minimize our loss function, so we want to go downhill. If you release a boulder on the mountain side, where will it go? Downhill. Will it go in a straight line? No. It will take the path with the greatest descent.

In the Gradient Descent method, we find the gradient, then go in the downhill direction. That is, we find how to change our variables to increase the function and we go the other way.

We calculate the gradient as,
$$\nabla f(\vec{x}) = \frac{\partial f}{\partial x}\hat{x} + \frac{\partial f}{\partial y}\hat{y}$$

Very simple example that we learned in calculus 1. Let $f(x) = 3(x+4)^2$. What is the gradient, and when is that gradient equal to 0?
$$\nabla f(\vec{x}) = \frac{\partial}{\partial x} 3(x+4)^2 = 6(x+4)$$
$$\nabla f(\vec{x}) = 6(x+4) = 0 \qquad\to\qquad x=-4$$

> Graph it and find where the slope is 0 and where the slope is steepest

So, the gradient is just a multi-dimensional version of finding the slope using the derivative. Now, a more complicated example. Find the gradient of the following function and find where the gradient is 0.
$$f(x,y) = \frac{x^2}{3} + x^4 + \frac{y^2}{5}$$
$$\nabla f(x,y) = \frac{\partial f}{\partial x} + \frac{\partial f}{\partial y} = \frac{2}{3}x + 4x^3 + \frac{2}{5}y$$
$$\nabla f(x,y) = \frac{2}{3}x + 4x^3 + \frac{2}{5}y = 0 \qquad\to\qquad y = -\frac{5}{3}x + 10x^3
$$\nabla f(x,y) = 11x + 9y = 0 \qquad \to \qquad y = -\frac{11}{9}x$$

> Open in https://www.desmos.com/3d and show $z = 3x^2 + 5xy + 2y^2$ and $y = -\tfrac{11}{9}x$
> https://www.desmos.com/3d/p3lywcgj7r to see the gradient

Usually, it's not so easy to find the minimum. So, we become the boulder. That is, we use the gradient to point us in the downhill direction, and then step that direction, then do an iteration until we stop going downhill.
$$\vec{x}_{i+1} = \vec{x}_i - \nabla f(\vec{x})$$

### Learning rate
Unfortunately, if we make a sudden change to our minimum point, then we may overshoot our goal. To prevent this, we take small "steps". The size of our step is referred to as the __learning rate__. 
$$\vec{x}_{i+1} = \vec{x}_i - \delta \nabla f(\vec{x})$$

What happens if the learning rate is too small?

What happens if the learning rate is too big?

What if you hit a plateau?

What if you hit a local minimum, but there is another global minimum?
* Fortunately, some loss functions (including MSE) are convex functions, meaning it only has one global minimum and no local minima

### Apply to Least Squares Regression

> ./Code/17_GradientDescent.py

### Stochastic Gradient Descent
