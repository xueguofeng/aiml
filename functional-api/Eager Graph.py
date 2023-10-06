import tensorflow as tf # Our main TensorFlow import
import timeit # Timeit module provides a simple way to time small bits of Python code.

def eager_function(x):
  result = x ** 2
  return result

x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])

print("The Eager Execution Mode:")
temp = eager_function(x)
print(temp)

print("The Graph Execution Mode:")
graph_function = tf.function(eager_function) # tf.function() converts eagertensor to graph function
temp = graph_function(x)
print(temp)
