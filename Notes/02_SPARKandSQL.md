<head>
<title>SQL and SPARK</title>
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

Once MapReduce showed how effective it was, many extensions were added to make it more powerful. For example, extensions were added to handle "workflow" systems. These allow more complicated connections than simply Map nodes feeding Reduce nodes. (See Leskovec Figure 2.6)

Two very common systems that follow the workflow system are *Spark* and *TensorFlow*.

## Spark
Spark is a workflow system and an adaptation of the MapReduce algorithm. In addition to the functionality of MapReduce, Spark introduces,
1. a more efficient way of coping with failures,
2. a more efficient way of grouping tasks, and
3. integration of programming language features such as looping and function libraries.

Spark's central data abstraction is a *Resilient Distributed Dataset* (RDD). An RDD is a file of objects of one type, though the type is not restricted as it would be in MapReduce.
* Distributed as the dataset can be broken apart and sent to different worker nodes
* Resilitent as the dataset is expected to be recovered at the end

Spark has two main parts in its execution:
1. Create Session
2. Data Logic

### Create Session
1. The driver is set up
2. The cluster manager assigns worker nodes to the task

### Data Logic
Spark is a lazy system, meaning that instead of doing all commands at once, it collects all commands then, when it hits an action method, it finds the most efficient way to execute all those commands.

```Python
# Create Session
Spark = SparkSession.builder.getOrCreate()

# Data Logic
df = spark.read.table('Sales')           # No action yet
df = df.filter('amount > 100')           # No action yet
df = df.groupBy('country').sum('Sales')  # No action yet
df = df.show()                           # ACTION - Execute all commands
```

At this point, the driver separates data into partitions, then assigns each partition to a worker. When completed, a combiner find the result, which the driver then outputs.

## PySpark
You can run the data through Databricks which will greatly simplify the process. If you want to install and run it locally,
* Install Java Development Kit (JDK) 
  * For Apache Spark 3.3 - 3.5, you need JDK 8, 11, or 17
  * For Apache Spark 4.x, you need JDK 17, 21, or 25.
* Install Apache Spark 
* Install Python and Jupyter Lab
* Launch Jupyter Lab




## SQL and Postgre SQL
One of the most traditional programming languages for databases is SQL (Structured Query Language). Most data handlers should know SQL. Python has simplified the process, but SQL is still a much faster and more powerful tool.

We can utilize SQL directly in python code.
* [Examples of SQL in Python](https://github.com/drolsonmi/math3280/blob/master/Notes/Code/02_SQL.ipynb)

We aren't focusing on SQL in this class - we'll stick with Python. But it is important to see that you can do SQL commands within python.