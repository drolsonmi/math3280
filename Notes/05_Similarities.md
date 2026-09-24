<head>
<title>Similarities</title>
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

# Similarities

## Shingling of Documents

If we want to compare two documents, we could compare every character and sequence of characters. However, this is a very costly process in computational resources and in time. A more efficient method is to look instead at sets of short strings that appear in documents.

A __shingle__ is a string of characters within a phrase. A __$k$-shingle__ is a shingle of size $k$.

Take the phrase: "Data Mining". The k-shingles are (Using the "_" to indicate the space between the words):
* $k=2$ : {Da, at, ta, a_, _M, Mi, in, ni, in, ng}$
* $k=3$ : {Dat, ata, ta_, a_M, _Mi, Min, ini, nin, ing}$
* $k=4$ : {Data, ata_, ta_M, a_Mi, _Min, Mini, inin, ning}$
* $k=5$ : {Data_, ata_M, ta_Mi, a_Min, _Mini, Minin, ining}$
* $k=6$ : {Data_M, ata_Mi, ta_Min, a_Mini, _Minin, Mining}$
* $k=7$ : {Data_Mi, ata_Min, ta_Mini, a_Minin, _Mining}$
* $k=8$ : {Data_Min, ata_Mini, ta_Minin, a_Mining}$
* $k=9$ : {Data_Mini, ata_Minin, ta_Mining}$
* $k=10$ : {Data_Minin, ata_Mining}$
* $k=11$ : {Data_Mining}$

Once a word/phrase/document is broken into $k$-shingles, then we just compare the number of similar shingles.

*Example*: What is the $k=2$ shingle Jaccard Similarity between the names "John" and "Joan"?
* John: $\{Jo, oh, hn\}$
* Joan: $\{Jo, oa, an\}$
$$[oh, hn, (Jo], oa, an)$$
$$SIM(John,Joan) = \frac{|John \cap Joan|}{|John \cup Joan|} = \frac{1}{5}$$

What size should we pick for $k$?
* If it's too small, then we risk having sequences that frequently appear in many documents, giving a high Jaccard similarity even though they are completely different
  * $k$ should be picked large enough that the probability of any given shingle appearing in any given document is low
* If it's too large, then we end up with very sparse data - too many possibilities that have almost no chance of occuring - resulting in large nonessential computations

A couple rules of thumb we can follow:
* Including a space for any whitespace, there are 27 possible characters, so there are $27^k$ possible shingles
* Consider an email
  * Using a $k=2$ shingle, there are $27^2 = 729$ different possible shingles. This is about the size of a typical email, so the probability for each of these shingles occuring is relatively high.
  * Using a $k=5$ shingle, there are $27^5 = 14,348,907$ different possible shingles. This is far larger than the size of a single email, so the probability for each of these shingle occuring is relatively low.
* Some characters are more likely to occur than others (*a* is more common than *z*)
  * Good rule of thumb is to approximate with only 20 possible characters, so the number of $k$-shingles is __$20^k$__

For an email, $k=5$ is pretty good. For a large document (such as a research paper), we may want to increase that to $k=9$.

To simplify calculations, instead of keeping a list of possible shingles and trying to match, we can use *hash functions*.
* A $k=9$ shingle takes up 9 bytes (one for each character)
* Using a hash function between $0$ and $2^{32} - 1$, each shingle can be represented using only 4 bytes
  * 1 byte = 8 bits, so 4 bytes = 32 bits

### Shingles built from Words
Although this does simplify the process of tracking the shingles, it is still too sparse. That is, there are still too many unused shingles. Instead, we can create *shingles built from words.* A common method is to use stop-word-based shingles (the, a, an, for, that, have, ...)

GEICO advertisement
* 15 minutes could save you 15% or more on car insurance
* Has 4 stop-word shingles:
  * could save you
  * you 15% or
  * or more on
  * on car insurance
  
GEICO statement
* If you own a car, you need auto insurance. A car insurance policy helps provide financial protection for you, and possibly others if you're involved in an accident. All it takes is a few minutes to get a personalized car insurance quote with the best rates, discounts, and options available to you.
* Has 17 stop-word shingles:
  * if you own
  * a car you
  * a car insurance
  * for you and
  * and possibly others
  * if you're involved
  * in an accident
  * an accident all
  * it takes is
  * is a few
  * a few minutes
  * to get a
  * a personalized car
  * with the best
  * the best rates
  * and options available
  * to you
  

## Similarity-Preserving Summaries of Sets
* Shingling takes a lot of space - nearly 4x as much as the document itself
* Large numbers of documents could make it impossible to store all the shingle-sets in memory
* Instead of sets, we're going to replace them with "__signatures__"
  * This won't give exact Jaccard similarities, but it gives a good estimate
  * The larger the signatures, the more accurate the estimates become

We'll look at these signatures by,
1. Representing the sets as a matrix
2. Use Minhashing to get a signature from each set

Let's define a few sets:
$$S_1 = \{a,c,d\}\qquad S_2 = \{b,c,e\} \qquad S_3 = \{a,e\} \qquad S_4 = \{c,d\}$$

Now, we'll represent each set as a column in a matrix $A$, and the elements will be set as rows in that matrix. We'll set $A_{ij}=1$ if element $i$ is in set $S_j$. 

| Element |  S1   |  S2   |  S3   |  S4   |
| :-----: | :---: | :---: | :---: | :---: |
|    a    |   1   |   0   |   1   |   0   |
|    b    |   0   |   1   |   0   |   0   |
|    c    |   1   |   1   |   0   |   1   |
|    d    |   1   |   0   |   0   |   1   |
|    e    |   0   |   1   |   1   |   0   |

An example of this might be that each row is a product from a store, and each column a customer, then each cell would represent whether the customer bought (1) that product or not (0).

This isn't how data is stored, but it's a first step in helping us understand the process. A matrix like this would be sparse.
* Instead of storing the data, we store the location of any 1
* Information would be stored instead as a list of tuples containing the element and the set
  * Using the example of purchases, a tuple might contain (item, purchaser, [details])


## Minhashing
Datasets can be very large, and calculating the Jaccard Similarity can be resource-intensive. With a __minhash__, we find a signature of entries to represent the original data, thus decreasing the resources needed to calculate $SIM(S_i,S_j)$. There are three steps to calculating the minhash:
1. Permute rows (or rather, randomize the indices)
2. Select the first entry in the permuted data to represent the original data
3. Repeat this process $n$ times to create a __signature__.

First, we permute the rows:

| Element |  S1   |  S2   |  S3   |  S4   |
| :-----: | :---: | :---: | :---: | :---: |
|    e    |   0   |   1   |   1   |   0   |
|    b    |   0   |   1   |   0   |   0   |
|    a    |   1   |   0   |   1   |   0   |
|    c    |   1   |   1   |   0   |   1   |
|    d    |   1   |   0   |   0   |   1   |

Second, we choose the first value that appears in each set.

|  S1   |  S2   |  S3   |  S4   |
| :---: | :---: | :---: | :---: |
|  a    |  e    |  e    |  c    |

Then, we repeat.
> | Element |  S1   |  S2   |  S3   |  S4   |
> | :-----: | :---: | :---: | :---: | :---: |
> |    c    |   1   |   1   |   0   |   1   |
> |    b    |   0   |   1   |   0   |   0   |
> |    d    |   1   |   0   |   0   |   1   |
> |    a    |   1   |   0   |   1   |   0   |
> |    e    |   0   |   1   |   1   |   0   |
> 
> |  S1   |  S2   |  S3   |  S4   |
> | :---: | :---: | :---: | :---: |
> |  a    |  e    |  e    |  c    |
> |  c    |  c    |  a    |  c    |

And we repeat 100+ times. As you can tell, this process can get very obnoxious if we write a matrix for each repetition. It is worse to code it this way. So, we'll instead randomize the elements (or maybe the indices) and find the first element in the new order that appears in each set.
  * Random Order: ebacd
  * S1: e b __a c d__ --> a
  * S2: __e b__ a __c__ d --> e
  * S3: __e__ b __a__ c d --> e
  * S4: e b a __c d__ --> c

Then repeat this 100+ times.
> | Random Order |  S1   |  S2   |  S3   |  S4   |
> | :----------: | :---: | :---: | :---: | :---: |
> | ebacd        |   a   |   e   |   e   |   c   |
> | cbdae        |   c   |   c   |   a   |   c   |
> | cdbea        |   c   |   c   |   e   |   c   |
> | bdaec        |   d   |   b   |   a   |   d   |
> | becad        |   c   |   b   |   e   |   c   |
> | dcbae        |   d   |   c   |   a   |   d   |
> | acebd        |   a   |   c   |   a   |   c   |

The column is then the signature for that set, and we find the similarity between two signatures.

$$J(S_1,S_2) = \frac{\#~of~matches}{\#~of~pairs} = \frac{|(c,c),(c,c)|}{|(a,e),(c,c),(c,c),(d,b),(c,b),(d,c),(a,c)|} = \frac{2}{7}$$

$$J(S_1,S_3) = \frac{1}{7} \qquad J(S_1,S_4) = \frac{5}{7}$$

$$J(S_2,S_3) = \frac{1}{7} \qquad J(S_2,S_4) = \frac{3}{7} \qquad J(S_3,S_4) = \frac{0}{7}$$

See the [Similarities iPython Notebook](./Code/05_Similarities.ipynb) to see examples of how this is done.

The interesting thing about Minhashing is that the probability that any two sets has the same hash function equals the Jaccard similarity between those sets.

The math of this relationship:
* When looking at any row in any two sets, there are three possibilities:
    1. Both values are a 1 (We'll call this a type $X$ row)
    2. One value is 1 and the other is 0 (We'll call this a type $Y$ row)
    3. Both values are a 0 (We'll call this a type $Z$ row)
       * In a Venn diagram, the intesection would be group $X$, the rest of the area inside the dataset is group $Y$, and the area outside the diagram is $Z$
         
* Let $x$ be the number of rows of type $X$, or the number of rows where $S_1$ and $S_2$ are both 1
  * $\lvert S_1 \cap S_2\lvert = x$
* Let $y$ be the number of rows of type $Y$, or the number of rows where either $S_1$ or $S_2$ is 1, but not both (that is, everything *except* the intersection)
  * $\lvert S_1 \cancel{\cap} S_2\lvert = y$
  * $\lvert S_1 \cup S_2\lvert = \lvert S_1 \cap S_2\lvert + \lvert S_1 \cancel{\cap} S_2\lvert = x+y$
* The Jaccard similarity would be:
$$J(S_1,S_2) = \frac{|S_1 \cap S_2|}{|S_1 \cup S_2|} = \frac{x}{x+y}$$

$$P\left[h(S_1) = h(S_2)\right] = \frac{\#~times~S_1=S_2}{Size~of~S1~and~S2} = \frac{x}{x+y} = J(S_1,S_2)$$

* The Jaccard similarity between $S_i$ and $S_j$ is estimated by the probability that the two columns have the same value in a given row of the signature matrix
* The expected number of rows in which two columns agree equals the Jaccard similarity of their corresponding sets
* The more rows in the signature matrix, the smaller the expected error in the estimate of the Jaccard Similarity will be

See the [Similarities iPython Notebook](./Code/05_Similarities.ipynb) for calculation of Jaccard Distance.

### Computing Minhash Signatures in Practice
As we have mentioned, we can't actually permute the rows in real life as that would be very resource intensive. Instead, we follow this algorithm for each signature:
1. Use a hash function on the index to find a new value
     * Typical to do a calculate and find the modulus with the number of rows (length of each individual dataset)
     * Works best when there is a prime number of rows - no repeated hash values
2. Find the signature
     * Set signature for all columns to a large number
     * For each column $c$:
         * If $c$ has a 0 in row $r$, signature remains unchanged
         * If $c$ has a 1 in row $r$ but hash value is larger than current signature, signature remains unchanged
         * If $c$ has a 1 in row $r$ and hash value is smaller than current signature, signature is replaced with hash value
  
Example:

| Row |   S1  |   S2  |   S3  |   S4  | $h_1(x) = x+1~mod~5$ | $h_2(x) = 3x+1~mod~5$ | $h_3(x) = 3x+2~mod~6$ | $h_4(x) = x-2~mod~7$ |
| --- | :---: | :---: | :---: | :---: | :------------------: | :-------------------: | :-------------------: | :------------------: |
|  0  |   1   |   0   |   0   |   1   |          1           |           1           |           2           |          5           |
|  1  |   0   |   0   |   1   |   0   |          2           |           4           |           5           |          6           |
|  2  |   0   |   1   |   0   |   1   |          3           |           2           |           2           |          0           |
|  3  |   1   |   0   |   1   |   1   |          4           |           0           |           5           |          1           |
|  4  |   0   |   0   |   1   |   0   |          0           |           3           |           2           |          2           |

Obviously the third hash function has flaws. We'll include it, but only to show a point at the end.

Here are the Jaccard Similarities for our 4 sets. We'll compare our final calculations to these similarities.

|       | S1    | S2            | S3            | S4            |
| :---: | :---: | :-----------: | :-----------: | :-----------: |
|  S1   |       | $\frac{0}{3}$ | $\frac{1}{4}$ | $\frac{2}{3}$ |
|  S2   |       |               | $\frac{0}{4}$ | $\frac{1}{3}$ |
|  S3   |       |               |               | $\frac{1}{5}$ |
|  S4   |       |               |               |               |

We'll set up our signature matrix with a value larger than the possible hash values (since the hash function is at most $mod~6$, the largest value we can get is 5. So, let's set all initial signatures to 9.

| Hash Function |  S1   |  S2   |  S3   |  S4   |
| :-----------: | :---: | :---: | :---: | :---: |
|     $h_1$     |   9   |   9   |   9   |   9   |
|     $h_2$     |   9   |   9   |   9   |   9   |
|     $h_3$     |   9   |   9   |   9   |   9   |
|     $h_4$     |   9   |   9   |   9   |   9   |

Then for each row, test to see if values need to be replaced. 

Row 0:
* S1 and S4 have 1's, so replace {9,9,9,9} with {1,1,2,5}
  * To simplify this, let $c = min(a,b)$ where $c_i = min(a_i,b_i)$ for $i=1,...,n$
  * min({9,9,9,9}, {1,1,2,5}) = {1,1,2,5})

| Hash Function |  S1   |  S2   |  S3   |  S4   |
| :-----------: | :---: | :---: | :---: | :---: |
|     $h_1$     |   1   |   9   |   9   |   1   |
|     $h_2$     |   1   |   9   |   9   |   1   |
|     $h_3$     |   2   |   9   |   9   |   2   |
|     $h_4$     |   5   |   9   |   9   |   5   |

Row 1:
* S3 has a 1, so replace {9,9,9,9} with {2,4,5,6} (min({9,9,9,9}, {2,4,5,6}) = {2,4,5,6})

| Hash Function |  S1   |  S2   |  S3   |  S4   |
| :-----------: | :---: | :---: | :---: | :---: |
|     $h_1$     |   1   |   9   |   2   |   1   |
|     $h_2$     |   1   |   9   |   4   |   1   |
|     $h_3$     |   2   |   9   |   5   |   2   |
|     $h_4$     |   5   |   9   |   6   |   5   |

Row 2:
* S2 has a 1, min({9,9,9,9}, {3,2,2,0}) = {3,2,2,0}
* S4 has a 1, min({1,1,2,5}, {3,2,2,0}) = {1,1,2,0}

| Hash Function |  S1   |  S2   |  S3   |  S4   |
| :-----------: | :---: | :---: | :---: | :---: |
|     $h_1$     |   1   |   3   |   2   |   1   |
|     $h_2$     |   1   |   2   |   4   |   1   |
|     $h_3$     |   2   |   2   |   5   |   2   |
|     $h_4$     |   5   |   0   |   6   |   0   |

Row 3: 
* S1 is a 1. min({1,1,2,5}, {4,0,5,1}) = {1,0,2,1}
* S3 is a 1. min({2,4,5,6}, {4,0,5,1}) = {2,0,5,1}
* S4 is a 1. min({1,1,2,0}, {4,0,5,1}) = {1,0,2,0}

| Hash Function |  S1   |  S2   |  S3   |  S4   |
| :-----------: | :---: | :---: | :---: | :---: |
|     $h_1$     |   1   |   3   |   2   |   1   |
|     $h_2$     |   0   |   2   |   0   |   0   |
|     $h_3$     |   2   |   2   |   5   |   2   |
|     $h_4$     |   1   |   0   |   1   |   0   |

Row 4: 
* S3 is a 1. min({2,0,5,1}, {0,3,2,2}) = {0,0,2,1}

| Hash Function |  S1   |  S2   |  S3   |  S4   |
| :-----------: | :---: | :---: | :---: | :---: |
|     $h_1$     |   1   |   3   |   0   |   1   |
|     $h_2$     |   0   |   2   |   0   |   0   |
|     $h_3$     |   2   |   2   |   2   |   2   |
|     $h_4$     |   1   |   0   |   1   |   0   |

Similarities from minhashing:

|       | S1            | S2            | S3            | S4            |
| :---: | :-----------: | :-----------: | :-----------: | :-----------: |
|  S1   |               | $\frac{0}{3}$ | $\frac{1}{4}$ | $\frac{2}{3}$ |
|  S2   | $\frac{1}{4}$ |               | $\frac{0}{4}$ | $\frac{1}{3}$ |
|  S3   | $\frac{3}{4}$ | $\frac{1}{4}$ |               | $\frac{1}{5}$ |
|  S4   | $\frac{3}{4}$ | $\frac{2}{4}$ | $\frac{2}{4}$ |               |

Obviously not the same as before. But if we continue the minhashing process a very large number of times and eliminate hash functions that don't give true permutations, then the similarities will eventually approach the true Jaccard Similarity.

Important Note:
* Note how $J(S1,S2) = 0$ is the actual similarity. How is it possible that we got a similarity $\ne 0$?
    * This came because of $h_3(x)$. Remember that this was a bad hash function. If we had avoided it, then this wouldn't have happened.

