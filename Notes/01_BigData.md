<head>
<title>Big Data</title>
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

# Big Data

## Cluster Computing
When dealing with large computations, such as large-scale models, one computer will not be enough.
> As an undergraduate, I created a model of air pollution in North Salt Lake. It was a 24-hour model covering a 100-km^2 area. It took well over 1 hour on my computer to get the results. Imagine how much more time it would have taken as a 3-Dimensional model covering the entire planet... By the time my computer finished a forecast model, the event being forecasted would have happened weeks ago.

To handle large computers, we utilize __parallel processing__, where several processors are linked together and work on parts of the problem simultaneously. This helps the calculations to complete in far less time.
* Each processor is called a __node__.
* The collection of nodes is called a __supercomputer__.

In data science, however, we are not only dealing with large computations, but with large amounts of data as well. 
* For example, large-scale Web services, such as Google or Amazon, are continually dealing with large amounts of data and customer interactions.

To handle this, we use not only the processors on each node, but the storage space as well. 
* These systems are known as __computing clusters__.
* The software to manage the data and queries is a __distributed file system__.


* Each processor/storage unit is called a __node__.
* Each node is installed on a __rack__.
  * There are often 8-64 nodes on a rack.
  * Each node on that rack is connected by a localized network - typically a gigabit ethernet.
* A collection of several racks is a __cluster__.
  * Several racks are then connected by another level of network or a switch.

### Problems with Computing Clusters
In order to get all the information from the racks to work with each other, they need more bandwidth than the rack itself has. We will learn about how these are used soon. First, let's look at the hardware challenges.

All hardware eventually fails. With heavy usage, it will fail faster. 
* In large-scale services, one node can last about 3 years (a little more than 1000 days)
* If I have a server of 1000 nodes, that means that on average, 1 node will fail every day
* A server at Google may have a million nodes, which means there are about 1000 nodes that fail every day

With so many failures, we have to ensure no disruption in data or in calculations if the failure happens while the program is running. To ensure this happens, there are two requirements:
1. Files must be stored redundantly
2. Computations must be divided into smaller tasks
    * If one task fails, then only that one task needs to be restarted, not the entire program

### Distributed File System Organization
A __distributed file system (DFS)__ works by dividing the data file into separate pieces and copying them.
1. Files are divided into __chunks__, typically 64 MB
    * Size can be determined by the user
2. Each chunk is the replicated and saved on different nodes
    * Number of copies can be determined by the user
    * The nodes holding the copies should be on different racks so copies aren't all lost if a rack fails
3. A __master node__ (or name node) tracks the location of all chunks so retrieval is simplified
    * The master node is also replicated

A DFS is often used when,
* individual files are large (terabytes), and
* files are rarely updated

There is no need for files to be distributed if they are small. And if any file is frequently updated, then the process becomes very complicated. So, this may be a good system for data on a global scale, but wouldn't work well for Amazon who has changes in inventory and prices daily.

### MapReduce
__MapReduce__ is the style of computing that is used to implement the DFS methodology. There are many different implementations:
* (GFS) Google File System - The original
* (HDFS) Hadoop DFS - Open-source, distributed by the Apache Software Foundation
* Spark
* Colossus - An improved version of GFS

The MapReduce process only involves two functions: *Map* and *Reduce*. The process is as follows. We'll follow the process with an example of counting the number of words.
1. *Map tasks* are given one or more chunks from the DFS and matches it into key-value pairs
    * Each map task looks for the words $w_1$, $w_2$, etc.
    * The key-value pair would be <$w_1, v_1$>, <$w_2,v_2$>, etc.
    * The result is a list of all key-value pairs <$w_i, v_i$> for all documents
2. A __master controller__ sorts and groups these key-value pairs and assigns them to a *Reduce task* 
    * All word pairs are sorted: <$w_1, v_1$>, <$w_1, v_x$>, ..., <$w_2, v_2$>, <$w_2, v_y$>, ..., <$w_3, v_3$>, <$w_3, v_z$>, ... 
    * These pairs are then grouped as <$w_1, [v_1,...]>$, <$w_2,[v_2,...]$>, <$w_3, [v_3,...]$>, ... ,
    * Each group is then assigned to a *Reduce Task* for the final computation
3. *Reduce Tasks* work with one key at a time, combining the values associated with that key in some way
    * All pairs with $w_1$ are given to one reduce task, $w_2$ to another, etc.
    * If the tasks are small enough, multiple tasks can be assigned to the same node
      * Input to ReduceTask1: <$w_1, [v_1,...]$>
      * Input to ReduceTask2: <$w_2, [v_2,...]$>
      * Input to ReduceTask3: <$w_3, [v_3,...]$>, <$w_4, [v_4, ...]$>
      * ...
    * Combine all the values together using a __combiner__ (a reduce function that is associative and commutative)
      * Output from ReduceTask1: <$w_1, x_1$>
      * Output from ReduceTask2: <$w_2, x_2$>
      * Output from ReduceTask3: <$w_3, x_3$>, <$w_4, x_4$>

The Master Controller handles the process by,
1. Assigning nodes in the cluster to complete either a Map Task or a Reduce Task, never both
    * Nodes assigned to complete Map or Reduce Tasks are known as __workers__
2. Tracks the status of workers
    * When workers report that they are done, the Master Controller can schedule a new task to that node

Example of a MapReduce process given in PowerPoint.

#### Linear Algebra example on MapReduce
The original use for MapReduce was to complete Matrix-Vector multiplication. We will look at the calculation of $M\vec{v}=\vec{x}$ where $M$ is a $p\times q$ matrix, $\vec{v}$ is a vector with $q$ elements, and $\vec{x}$ is the result of the calculation, a vector with $p$ elements.

$$\begin{bmatrix}
  m_{00} & m_{01} & m_{02} & \dots  & m_{0q} \\
  m_{10} & m_{11} & m_{12} & \dots  & m_{1q} \\
  m_{20} & m_{21} & m_{22} & \dots  & m_{2q} \\
  \vdots &        &        & \ddots & \vdots \\
  m_{p0} & m_{p1} & m_{p2} & \dots  & m_{pq} \\
\end{bmatrix}\begin{bmatrix}
  v_0 \\ v_1 \\ v_2 \\ \vdots \\ v_q
\end{bmatrix} = \begin{bmatrix}
  x_0 \\ x_1 \\ \vdots \\ x_p
\end{bmatrix}$$

Note that MapReduce is not helpful when $p$ is small enough ($p=100$) to be done on individual computers. MapReduce is more useful when $M$ is so large that it doesn't fit into the memory of a single node.

#### Matrix-Vector Multiplication with small vectors
Look first at the case when $q$ is small enough that $\vec{v}$ fits into memory. But $M$ is still too large.
* Divide $M$ into sections with multiple rows

$$\begin{bmatrix}
  m_{00} & m_{01} & m_{02} & \dots  & m_{0q} \\
  m_{10} & m_{11} & m_{12} & \dots  & m_{1q} \\
  m_{20} & m_{21} & m_{22} & \dots  & m_{2q} \\
  ---    & ---    & ---    & ---    & ---    \\
  m_{30} & m_{31} & m_{32} & \dots  & m_{3q} \\
  m_{40} & m_{41} & m_{42} & \dots  & m_{4q} \\
  m_{50} & m_{51} & m_{52} & \dots  & m_{5q} \\
  ---    & ---    & ---    & ---    & ---    \\
  m_{60} & m_{61} & m_{62} & \dots  & m_{6q} \\
  m_{70} & m_{71} & m_{72} & \dots  & m_{7q} \\
  m_{80} & m_{81} & m_{82} & \dots  & m_{8q} \\
  ---    & ---    & ---    & ---    & ---    \\
  \vdots &        &        & \ddots & \vdots \\
  m_{p0} & m_{p1} & m_{p2} & \dots  & m_{pq} \\
\end{bmatrix}\begin{bmatrix}
  v_0 \\ v_1 \\ v_2 \\ \vdots \\ v_q
\end{bmatrix}$$

* The vector $\vec{v}$ is stored in the memory of each Map worker
* The first Map worker is assigned this first section, and so on
* Map Task: Make a list of all key value pairs <$i,m_{ij}, v_j$>
  * <$0,m_{00},v_0$>, <$0,m_{01},v_1$>, <$0,m_{02},v_2$>, ..., <$0,m_{0q},v_q$>
  * <$1,m_{10},v_0$>, <$1,m_{11},v_1$>, <$1,m_{12},v_2$>, ..., <$1,m_{1q},v_q$>
  * ...
  * <$p,m_{p0},v_0$>, <$p,m_{p1},v_1$>, <$p,m_{p2},v_2$>, ..., <$p,m_{pq},v_q$>
* Grouping: Take the product $m_{ij}v_j$ and group the results with the same index $i$
  * <$0, [m_{00}v_0, m_{01}v_1, m_{02}v_2, ... , m_{0q}v_q]$>
  * <$1, [m_{10}v_0, m_{11}v_1, m_{12}v_2, ... , m_{1q}v_q]$>
  * <$2, [m_{20}v_0, m_{21}v_1, m_{22}v_2, ... , m_{2q}v_q]$>
  * ...
  * <$p, [m_{p0}v_0, m_{p1}v_1, m_{p2}v_2, ... , m_{pq}v_q]$>
* Reduce Task: Find the sum of all elements in each group
  * <$0, x_0$>
  * <$1, x_1$>
  * <$2, x_2$>
  * ...
  * <$p, x_p$>
    * where $x_i$ = $\sum_j m_{ij}v_j = m_{i0}v_0 + m{i1}v_1 + m{i2}v_2 + ...$
* Final Output:
$$\vec{x} = \begin{bmatrix}
  x_0 \\ x_1 \\ \vdots \\ x_p
\end{bmatrix}$$

#### Example
$$\begin{bmatrix}
2 &  8 & 14 \\
4 & 10 & 16 \\
6 & 12 & 18 
\end{bmatrix}\begin{bmatrix}
7 \\ 5 \\ 3
\end{bmatrix}$$

* Break each calculation into key-value pairs. We'll make the key the row number $i$, so the pairs will be $\left<i,m_{ij},v_j\right>$.
* Group the key-value pairs by the key, and assign it to the Reduce Task.
* Reduce tasks then combine the values, and a combiner sends the value through the function $f(i) = \sum_{j}m_{ij}v_j$.
$$\begin{matrix}
\left<0,2,7\right> \\
\left<0,8,5\right> \\
\left<0,14,3\right> \\
\left<1,4,7\right> \\
\left<1,10,5\right> \\
\left<1,16,3\right> \\
\left<2,6,7\right> \\
\left<2,12,5\right> \\
\left<2,18,3\right>
\end{matrix}\qquad \to \qquad\begin{bmatrix}
\left<0,2,7\right> \\
\left<0,8,5\right> \\
\left<0,14,3\right> \\
\\ \hline \\
\left<1,4,7\right> \\
\left<1,10,5\right> \\
\left<1,16,3\right> \\
\\ \hline \\
\left<2,6,7\right> \\
\left<2,12,5\right> \\
\left<2,18,3\right>
\end{bmatrix}\qquad\to\qquad\begin{matrix}
\left<0,[(2,7),(8,5),(14,3)]\right> ~~~\to~~~ \left<0,[14,40,42]\right> ~~~\to~~~ \mathbf{\left<0,96\right>} \\
\\ \hline\\
\left<1,[(4,7),(10,5),(16,3)]\right> ~~~\to~~~ \left<1,[28,50,48]\right> ~~~\to~~~ \mathbf{\left<1,126\right>} \\
\\ \hline \\
\left<2,[(6,7),(12,5),(18,3)]\right> ~~~\to~~~ \left<2,[42,60,54]\right> ~~~\to~~~ \mathbf{\left<2,156\right>}
\end{matrix}$$

#### Matrix-Vector Multiplication with large vectors
The previous works well when $q$ is small enough for $\vec{v}$ to fit in memory. But if $q$ is too large, then we can't do it as we did before. However, we can add one step which will allow us to continue this method: divide $\vec{v}$ into sections as well:

$$\begin{bmatrix}
  m_{00} & m_{01} & | & m_{02} & m_{03} & | & m_{04} & m_{05} & | & \dots  & m_{0q} \\
  m_{10} & m_{11} & | & m_{12} & m_{13} & | & m_{14} & m_{15} & | & \dots  & m_{1q} \\
  m_{20} & m_{21} & | & m_{22} & m_{23} & | & m_{24} & m_{25} & | & \dots  & m_{2q} \\
  ---    & ---    & | & ---    & ---    & | & ---    & ---    & | & ---    & ---    \\
  m_{30} & m_{31} & | & m_{32} & m_{33} & | & m_{34} & m_{35} & | & \dots  & m_{3q} \\
  m_{40} & m_{41} & | & m_{42} & m_{43} & | & m_{44} & m_{45} & | & \dots  & m_{4q} \\
  m_{50} & m_{51} & | & m_{52} & m_{53} & | & m_{54} & m_{55} & | & \dots  & m_{5q} \\
  ---    & ---    & | & ---    & ---    & | & ---    & ---    & | & ---    & ---    \\
  m_{60} & m_{61} & | & m_{62} & m_{63} & | & m_{64} & m_{65} & | & \dots  & m_{6q} \\
  m_{70} & m_{71} & | & m_{72} & m_{73} & | & m_{74} & m_{75} & | & \dots  & m_{7q} \\
  m_{80} & m_{81} & | & m_{82} & m_{83} & | & m_{84} & m_{85} & | & \dots  & m_{8q} \\
  ---    & ---    & | & ---    & ---    & | & ---    & ---    & | & ---    & ---    \\
  \vdots &        & | & \vdots &        & | & \vdots &        & | & \ddots & \vdots \\
  m_{p0} & m_{p1} & | & m_{p2} & m_{p3} & | & m_{p4} & m_{p5} & | & \dots  & m_{pq}
\end{bmatrix}\begin{bmatrix}
  v_0 \\ v_1 \\ --- \\ v_2 \\ v_3 \\ --- \\ v_4 \\ v_5 \\ --- \\ \vdots \\ v_q
\end{bmatrix}$$

* Map Task 1 will get the 1st 2 elements of rows 0-2 in $M$ and the 1st 2 elements of $\vec{v}$
* Map Task 2 will get the 2nd 2 elements of rows 0-2 in $M$ and the 2nd 2 elements of $\vec{v}$
* Map Task 3 will get the 3rd 2 elements of rows 0-2 in $M$ and the 3rd 2 elements of $\vec{v}$
* ...and so on until rows 0-2 are complete
* The next Map Task will get the 1st 2 elements of rows 3-5 in $M$ and the 1st 2 elements of $\vec{v}$
* The next Map Task will get the 2nd 2 elements of rows 3-5 in $M$ and the 2nd 2 elements of $\vec{v}$
* ...and so on...

Once the mapping task is done, the grouping and reduce tasks are the same as before.

#### MapReduce and Failed Nodes
What happens if a node fails in the middle?
* Best case scenario: only a single map task or reduce task needs to be restarted
* Worst cast scenario: the node at which the Master is executing fails, and the entire MapReduce job needs to be restarted.


