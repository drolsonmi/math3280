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

## Spark and PySpark
As our focus in this series of classes is on Machine Learning, we aren't going to focus on Spark or TensorFlow at this time.

## SQL and Postgre SQL
One of the most traditional programming languages for databases is SQL (Structured Query Language). Most data handlers should know SQL. Python has simplified the process, but SQL is still a much faster and more powerful tool.

We can utilize SQL directly in python code.
* [Examples of SQL in Python](https://github.com/drolsonmi/math3280/blob/master/Notes/Code/02_SQL.ipynb)

We aren't focusing on SQL in this class - we'll stick with Python. But it is important to see that you can do SQL commands within python.