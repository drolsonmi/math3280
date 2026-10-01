<head>
<title>Frequent Itemsets</title>
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

# Frequent Itemsets
We learned earlier in the semester about how similar two objects may be to each other. We'll turn now to association between two kinds of objects. That is, there may not be any similarity between two objects, but there may be a relationship between them. For example, milk and bread are not very similar at all, but they are frequently bought together.

A __market-basket__ model describes relationships between two kinds of objects. 
* *items*
* *baskets* (sometimes called *transactions*) consist of an *itemset*
    * We usually assume that the size of the itemset is much smaller than the total number of items

## Frequent
An itemset is said to be "frequent" if a subset of items appears in many baskets. Thinking of this mathematically, take an itemset $I$. 
* Define the __support__ of $I$ to be the number of baskets for which $I$ is a subset
* $I$ is frequent if $Support(I) > s$ where $s$ is the __support threshold__
    - If the total number of items is small, a threshold of 1/2 of the total number of baskets is reasonable
    - If the total number of items is large, the threshold might be smaller than 1% of the number of baskets

Consider the following sets of letters:
* {A, B, C}
* {B, C, F}
* {B, C, D}
* {C, D, E}
* {B, C, E}
* {A, C, D}
* {B, E, F}
* {B, C, E}

For this example, let's set our support threshold to $s = 4$.

A __singleton set__ is a set of just one item. The supports for all the singleton sets are:
$$Support(\{A\}) = 2 \qquad Support(\{B\}) = 6 \qquad Support(\{C\}) = 7 \qquad Support(\{D\}) = 3 \qquad Support(\{E\}) = 4 \qquad Support(\{F\}) = 2$$

From this, only items $B$ and $C$ are frequent, since they are greater than $s$.

Take the support for the __doubleton__ subset $I_1 = \{B, E\}$. If we have a support threshold of $s=4$, then $I_1$ is not frequent.

$$Support(I_1) = Support(\{B,E\}) = 3 < s$$

The subset $I_2 = \{B, C\}$ would be considered a frequent itemset, however.

$$Support(I_2) = Support(\{B,C\}) = 5 > 4$$

How can we use this information? Sometimes, the results of this calculation are useless. For example, the purchase of milk and eggs would be considered similar since they are often purchased together. However, hot dogs and mustard are not considered similar, but they would have a higher support. This opens a market tactic: offer a sale on hot dogs, but increase the price of mustard. When people buy hot dogs because they are on sale, they may say, "Oh, I need mustard," and they'll get it regardless of the price.

Common applications of frequent itemsets:
1. *Related concepts*: words that often appear in conjunction with a topic. For example, how often does the word "civil" come up in an article about "engineering"
2. *Plagiarism*: sentences that appear in different documents. A document that has a large number of sentences with high support may be indicative of a plagiarized document
3. *Biomarkers*: genes or proteins appear when exposed to certain deseases

## Association Rules
Now, we want to look at how often different subsets appear together. A common application of this would be recommendation systems ("customers who bought what is in your cart also bought [item]"). To indicate the association of basket $I$ with item $j$ as $I\to j$.

So, how likely will basket $I$ be associated with $j$? We'll measure this be defining the __confidence__ of $I\to j$ as,
$$Confidence(I\to j) = \frac{Support(I\cup \{j\})}{Support(I)}$$

Using our earlier example, how likely is the set $I = \{B, C\}$ to be associated with $\{E\}$?
$$Confidence(\{B,C\} \to \{E\}) = \frac{Support(\{B, C, E\})}{Support(\{B, C\})} = \frac{2}{5}$$

Another way to think of this: The set $\{B,C\}$ appears 5 times. Of those 5 times, $E$ appears twice.

The confidence is useful as long as $Support(I)$ is fairly large. However, the confidence means more when the association rule reflects a true relationship. So, we define the __interest__ of an association rule as the difference between the confidence and the fraction of baskets that contain $j$.
$$Interest(I\to j) = Confidence(I\to j) - \frac{Support(\{j\})}{\text{\# of baskets}}$$

The advantage to this is that if $I$ and $j$ aren't associated, then $Confidence(I\to j) \approx \frac{Support(\{j\})}{\text{\# of baskets}}$, so $Interest(\{B,C\}\to\{E\}) \approx 0$

Using our earlier example,
$$Interest(\{B,C\}\to\{E\}) = Confidence(\{B,C\} \to \{E\}) - \frac{Support(\{E\})}{8}$$
$$Interest(\{B,C\}\to\{E\}) = \frac{Support(\{B, C, E\})}{Support(\{B, C\})} - \frac{Support(\{E\})}{8} = \frac{2}{5} - \frac{4}{8} = -\frac{1}{10}$$

What do the numbers mean?
* If the interest is high, then $I$ has a high probability of causing $j$
* If the interest is highly negative, then $I$ has a high probability of discouraging $j$
* If the interest is near 0, then any association between $I$ and $j$ is likely coincidental

How does this compare with others? Find the Confidence and Interest of the following associations:
* $\{A,B\}\to\{C\}$
* $\{B,F\}\to\{E\}$
* $\{C,D\}\to\{A\}$
* $\{B,C\}\to\{F\}$
* $\{A,D\}\to\{E\}$
  
$$Confidence(\{A,B\}\to\{C\}) = \frac{1}{1} \qquad Interest(\{A,B\}\to\{C\}) = 1 - \frac{7}{8} = \frac{1}{8} = 0.125$$
$$Confidence(\{B,F\}\to\{E\}) = \frac{1}{2} \qquad Interest(\{B,F\}\to\{E\}) = \frac{1}{2} - \frac{4}{8} = 0\qquad\qquad$$
$$Confidence(\{C,D\}\to\{A\}) = \frac{1}{3} \qquad Interest(\{C,D\}\to\{A\}) = \frac{1}{3} - \frac{2}{8} = \frac{1}{12} = 0.0833$$
$$Confidence(\{B,C\}\to\{F\}) = \frac{1}{5} \qquad Interest(\{B,C\}\to\{F\}) = \frac{1}{4} - \frac{2}{8} = 0\qquad\qquad$$
$$Confidence(\{A,D\}\to\{E\}) = 0 \qquad Interest(\{A,D\}\to\{E\}) = 0 - \frac{4}{8} = -\frac{1}{2}\qquad\qquad$$

> Worked out example of Frequent Itemsets in [Itemsets iPython Notebook](https://github.com/drolsonmi/math3280/blob/master/Notes/Code/06_Itemsets.ipynb)

# Itemset Handling
## Calculations and Memory
For a small company, the calculation could easily be done on one computer. For a large company, the number of transactions (baskets) is so large that the file holding it can't fit into the memory of one computer, so it needs to be done with some sort of Distributed File System and use an algorithm such as MapReduce to find frequent itemsets. But this can be really hard. It is addressed in more detail at the end of the chapter.

Another possible problem that could take a lot of memory is the size of the baskets. Fortunately, most baskets are not that large. If we assume an average basket size of 20 items, then there are $\begin{pmatrix} 20 \\ 2 \end{pmatrix} = 190$ pairs of items. This can easily be done on a single computer.

|      | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 1    |      | x    | x    | x    | x    | x    | x    | x    | x    | x    |
| 2    | x    |      | x    | x    | x    | x    | x    | x    | x    | x    |
| 3    | x    | x    |      | x    | x    | x    | x    | x    | x    | x    |
| 4    | x    | x    | x    |      | x    | x    | x    | x    | x    | x    |
| 5    | x    | x    | x    | x    |      | x    | x    | x    | x    | x    |
| 6    | x    | x    | x    | x    | x    |      | x    | x    | x    | x    |
| 7    | x    | x    | x    | x    | x    | x    |      | x    | x    | x    |
| 8    | x    | x    | x    | x    | x    | x    | x    |      | x    | x    |
| 9    | x    | x    | x    | x    | x    | x    | x    | x    |      | x    |
| 10   | x    | x    | x    | x    | x    | x    | x    | x    | x    |      |

But that assumes we are doing pairs - only looking at 2 items at a time. If we want to look at larger subsets, the time for calculation increases. If it takes time $t$ to make one calculation, then the time it takes for the entire matrix is $n^k(t)$.
* If we ignore the cases comparing an itemset with itself, the time simplifies to $n(n-1)(n-2)\dots (t)$

$$(t)\prod_{i=0}^k (n-i)$$

* Usually we only need to deal with a subset of $k=2$ or $3$, so this usually doesn't become an issue
* In the rare cases where $k$ is large, there are often items in the basket that can be ignored and dropped, decreasing $n$.

Three methods to simplify this process are commonly used:
1. Triangular Matrix method
2. Triples method
3. A-Priori method

### Triangular Matrix method
Imagine a matrix of all items as rows and all items as columns. The values of the matrix would be the interest. However, note that the association between items A and B is nearly the same as between items B and A. So, let's just take the upper triangle of the matrix so that there are only values if $i<j$.

|      | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 1    |      | x    | x    | x    | x    | x    | x    | x    | x    | x    |
| 2    |      |      | x    | x    | x    | x    | x    | x    | x    | x    |
| 3    |      |      |      | x    | x    | x    | x    | x    | x    | x    |
| 4    |      |      |      |      | x    | x    | x    | x    | x    | x    |
| 5    |      |      |      |      |      | x    | x    | x    | x    | x    |
| 6    |      |      |      |      |      |      | x    | x    | x    | x    |
| 7    |      |      |      |      |      |      |      | x    | x    | x    |
| 8    |      |      |      |      |      |      |      |      | x    | x    |
| 9    |      |      |      |      |      |      |      |      |      | x    |
| 10   |      |      |      |      |      |      |      |      |      |      |

Once we have the triangular matrix, we can sort all values into an array, where the index $k$ is found by,
$$k=(i-1)\left(n-\frac{i}{2}\right) + j - i$$

where $1 \le i < j \le n$.
* Note that we do not count when anything is paired with itself. $i$ is strictly less than $j$.

> Worked out example of Triangular Matrix method in [Itemsets iPython Notebook](https://github.com/drolsonmi/math3280/blob/master/Notes/Code/06_Itemsets.ipynb)

Since we are only considering cases where $i < j$, this changes from a full (sparse) $n\times n$ matrix to the upper triangle above the diagonal (an $(n-1)\times(n-1)$ triangle). If we are dealing with 4-byte integers, then this drops us from a matrix taking up $4n^2$ bytes down to $4\cdot\frac{1}{2}(n-1)^2 = 2(n-1)^2$ bytes.

This is an improvement, but can still be sparse.

### Triples method
The advantage of the triples method is that we are only going to store information for each pair that *actually* occurs.

The triples method involves taking a triple $[i,j,c]$ where $c$ is the count for the pair $\{i,j\}$.
* Use a hash function with $i$ and $j$ to find where to store the data
* Store the values of $i$, $j$, and $c$ at that location

The advantage is that we are only storing data for each pair.

The disadvantage is that for each pair, we have to store 3 values. If we are working with 4-byte integers, then that is 12 bytes per pair.
* If more than 1/3 of possible pairs occur, the *triangular-matrix method* is better
* If less than 1/3 of possible pairs occur, the *triples method* is better

### A-Priori Algorithm
This method is not too difficult. However, as is always the problem in the real world, we are dealing with very large amounts of data that can't always be held in main memory, much less do calculations with them. So, to simplify the process, we'll look at the __A-Priori__ algorithm, which only looks at the most frequent items.

In order to understand the A-Priori algorithm, there are a few mathematical methods that need to be implemented. We won't go over these in detail here, but just give a quick summary. They are in section 6.2.3 - 6.2.4 of the *Leskovec* textbook.

#### Monotonicity
If a set $I$ of items is frequent, then so is every subset of $I$.

A couple of points:
* A subset $J\subseteq I$ has the same frequency if $J=I$
* J could be more frequent as $J$ could be a part of multiple subsets of $I$

Example:
* $I = \{milk,bread,eggs,chips,salsa\}$
* $J = \{milk,bread\}$
* If there are 20 people who bought $I$, then by default, the same 20 people also bought $J$
* Additionally, there are some people who did not get $I$, but still got $J$:
    * $\{milk, bread, eggs\}$
    * $\{milk, bread, chips\}$
    * $\{milk, bread, chips, salsa\}$
    * ...

#### The Algorithm
The basic principle of the A-Priori is to decrease the number of calculations. But in order to do this, we make 2 passes through the data instead of just 1.

__Pass 1__: Go through the data and count the frequency of each item
* Select only the items with a support over the support threshold (again, often around 1\%)
* Begin with a list of $n$ items, narrow that down to $m$ items, where $m$ is a small fraction of $n$.

__Pass 2__: For each basket, find only frequent items and count the pairs. Count all pairs for all baskets.

If 50\% of the items are eliminated, then since we are dealing with pairs, there will only be 25\% of the calculations. This saves a lot on time and memory.

In addition to being simple, this method can use monotonicity to find larger frequent $k$-sets.
* Let $C_k$ be the set of candidate $k$-sets from the 1st pass
* Find $L_k$, the set of truly frequent itemsets of size $k$
* Use $L_k$ to find candidates for the $(k+1)$-set ($C_{k+1}$)

From monotonicity, we know that there can only be the same (or fewer) number of frequent $(k+1)$-sets as there are frequent $k$-sets.
* Each set $L_k$ gets smaller as $k$ increases
* When $L_k$ is empty, then the largest frequent set is $L_{k-1}$

--------
## Homework
* Exercise 6.1.1 a,b,c
* Exercise 6.1.2
* Exercise 6.1.5 a,b - Find both the confidence and the interest of each association rule
* Exercise 6.1.7 a
* Exercise 6.2.6 a