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

With large datasets, a "reasonably high" interest would be items in 1\% of the baskets.

> Worked out example of Frequent Itemsets in [https://github.com/drolsonmi/math3280/blob/master/Notes/Code/06_Itemsets.ipynb]