<head>
<title>Linguistic Tools</title>
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

# Linguistic Tools

## Jaccard Similarity
The Jaccard Similarity compares two pieces of information to see how similar they are. Each row is a set
The calculation is,

$$J(S,T) = \frac{|S\cap T|}{|S\cup T|}$$

A simple example:

$$A = \{1, 3, 5\} \qquad B = \{3, 4, 5, 6\}$$

Venn diagram (Square brackets encompass elements of A, round brackets encompass elements of B):

$$\Big[1 \Big( 3, 5 \Big] 4, 6\Big)$$

There are 5 elements total, so $\left| A\cup B \right| = 5$. Only 2 elements are in both, so $\left| A\cup B \right| = 2$.

$$J(A,B) = \frac{|A\cap B|}{|A\cup B|} = \frac{2}{5}$$

There are two similarity calculations:
* Jaccard Similarity
  * Union is all elements, not repeated - just looking at possible values

$$\left| A\cup B \right| = \big|\{1, 3, 4, 5, 6\}\big| = 5 \qquad J(A,B) = \frac{2}{5}$$

* Jaccard Bag Similarity
  * Union is all elements in both sets combined, as if they were two bags mixed together

$$\left| A\cup B \right| = \big|\{1, 3, 5, 3, 4, 5, 6\}\big| = 7 \qquad J_B(A,B) = \frac{2}{7}$$

Example #2: You create a shopping list including,
* Milk (2), eggs, bread, chips (3), salsa

But you forget the shopping list. So, you get what you can remember, plus some additional things:
* Milk (3), eggs, chips (1), salsa, yogurt, cheese, ice cream

What is the Jaccard similarity?

$$\left| list \cap purchased \right| = \left| \{\text{milk, eggs, chips, salsa}\}\right|=4$$
$$\left| list \cup purchased \right| = \left| \{\text{milk, eggs, bread, chips, salsa, yogurt, cheese, ice cream}\}\right|=8$$
$$J(list, purchased) = \frac{\left| list \cap purchased \right|}{\left| list \cup purchased \right|} = \frac{4}{8}=0.5$$

Notice that we did not repeat milk or chips. For the Jaccard Similarity, we only consider similar items, not repeats. For the Jaccard Bag Similarity, we do consider repeats.
* For chips, it was on the list 3 times, but we only bought 1, so it is only counted once (1)
* For milk, it was bought 3 times, but only on the list 2 times, so there are only two (2) matched pairs
  * |list $\cap$ purchased|$ = |milk, milk, eggs, chips, salsa| = 5
* The union is all items, even if repeated
  * |list $\cup$ purchased|$ = |milk, milk, eggs, bread, chips, chips, chips, salsa, milk, milk, milk, eggs, chips, salsa, yogurt, cheese, ice cream| = 17

$$J_B(list, purchased) = \frac{\left| list \cap purchased \right|}{\left| list \cup purchased \right|} = \frac{5}{17}=0.294$$

Another example:

|     |  S  |  T  |
| --- | --- | --- |
| x_0 |  1  |  0  |
| x_1 |  0  |  1  |
| x_2 |  0  |  0  |
| x_3 |  1  |  1  |
| x_4 |  0  |  1  |
| x_5 |  1  |  0  |
| x_6 |  1  |  1  | 
| x_7 |  0  |  0  |
| x_8 |  1  |  1  |  
| x_9 |  0  |  1  |

To do this, we look at only positive results (entries with a "1"). The intersection would be where both $S$ and $T$ are 1:
$$\left| S\cap T \right| = 3$$

The union would be all entries where either $S$ or $T$ have a 1:
$$\left| S\cup T \right| = 8$$

We can consider, instead of a list of all datapoints, just count the number of all possibilities.

|  S  |  T  |  #  |
| --- | --- | --- |
|  0  |  0  |  2  |
|  0  |  1  |  3  |
|  1  |  0  |  2  |
|  1  |  1  |  3  |

or, looking at it with a confusion matrix,

|      |  S=1  |  S=0  |
| ---: | :---: | :---: |
|  T=1 |   3   |   3   |
|  T=0 |   2   |   2   |

$$\left| S\cap T \right| = 3 \qquad \left| S \cup T \right| = 3+3+2 = 8$$

Either way, the Jaccard Similarity is,
$$\left| S\cap T \right| = 3 \qquad \left| S\cup T \right| = 8 \qquad J(S,T) = \frac{\left| S\cap T \right|}{\left| S\cup T \right|} = \frac{3}{8}$$

The Jaccard Bag Similarity,
$$J_B(S,T) = \frac{3}{11}$$

The Jaccard Similarity can be used in a variety of ways:
* Similarity of Documents
* Plagiarism
* Mirror Pages
* Articles from the Same Source
* __Collaborative Filtering__
  * On-line Purchases
  * Movie Ratings

### Jaccard Distance
The Jaccard Similarity is a value between 0 (little to know similarity) and 1 (high similarity).

Sometimes we prefer using this value as a distance, but a distance of 0 means they are close and 1 means they are far apart. To get the Jaccard Distance, we take the complement of the Jaccard Similarity.
$$Jaccard~Distance = 1 - J(A,B)$$

## Regular Expressions (RegEx)

## TF-IDF
