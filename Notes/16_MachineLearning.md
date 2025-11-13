# 16 Machine Learning
__Math 3280 - Data Mining__ : Snow College : Dr. Michael E. Olson

### Additional Resources
* [YouTube: Steve Brunton - Intro to Data Science video playlist](https://www.youtube.com/watch?v=pzo13OPXZS4&list=PLMrJAkhIeNNQV7wi9r7Kut8liLFMWQOXn)
    * Specifically:
    * [Machine Learning Overview](https://www.youtube.com/watch?v=QpzQYDcczRQ)
    * [Types of Machine Learning 1](https://www.youtube.com/watch?v=YlGEQyEM_a8)
    * [Types of Machine Learning 2](https://www.youtube.com/watch?v=0_lKUPYEYyY)

-----


Three different categories of ML Algorithms
1. Supervision (does the data have labels or not?)
2. Online vs. Batch Learning (can it learn on the fly?)
3. Instance-based vs. Model-based Learning (how does it use new data?)

The criteria are not exclusive - all models can be described by each of these categories.
* We'll discuss supervision for the rest of the semester
* We'll address online/batch learning and instance/model-based learning next semester

## Supervision
* Supervised learning (Labelled Data)
  * Classification (Discrete Data)
  * Regression (Continuous Data)
* Unsupervised learning (Unlabelled Data)
  * Clustering (Discrete Data)
  * Visualization and Dimensionality Reduction (Continuous Data)
  * Anomaly Detection and Novelty Detection
  * Association Rule Learning
* Semisupervised learning (Partly automated, with some labels inserted at some point)

![Machine Learning Landscape](https://raw.githubusercontent.com/drolsonmi/math3480/main/Notes/Images/3480_ML_Landscape.png)

* System (Training Data)
* Labeled?
  * Yes: Supervised
    * Discrete or Continuous?
      * Discrete: Classification
      * Continuous: Regression
  * No: Unsupervised
    * Discrete or Continuous?
      * Discrete: Clustering
      * Continuous: Embedding (a.k.a. Dimensionality Reduction, Dimension Extraction, Pattern Extraction)
  * Partial: Semi-supervised
    * Model or Modify?
      * Model: Generative Models
      * Modify
        * --> Loops back to System

#### Supervised Learning Algorithms
* Classification
  * k-Nearest Neighbors (Also Unsupervised Clustering)
  * Support Vector Machines (SVMs)
  * Decision Trees and Random Forests
  * Neural Networks
* Regression
  * Linear Regression
  * Logistic Regression
  * Gaussian Processes

#### Unsupervised Learning Algorithms
* Clustering
  * K-Means
  * K nearest neighbors
  * Spectral Clustering
  * DBSCAN
  * Hierarchical Cluster Analysis (HCA)
* Dimensionality Reduction / Feature Extraction / Embedding / Visualization
  * Principal Component Analysis (PCA)
  * Autoencoder
  * Diffusion Maps
  
#### Semisupervised Learning Algorithms
* Model
  * Generative Models
      * Generative Adversarial Networks (GAN) - Two algorithms that fight with each other (supervise each other)
* Modify
  * Reinforcement Learning

