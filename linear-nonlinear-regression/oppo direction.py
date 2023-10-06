import numpy

# Z = x**2 + y**2, Z's largest value is 0, must find this largest value (global max)
# Z = -x**2 - y**2
# Calculate the minimum Z using Grdient Descent

x,y = 3,2 # Initial location
lr = 0.1 # learning rate

# result is a vector "gradient", the input is a Tuple "(2.0 * x, 2.0 * y)"
# Gradient, (∂z/∂x, ∂z/∂y)
gradient = numpy.array( (-2.0 * x, -2.0 * y) )

# Calculate the step to move, (⊿x, ⊿y)
# To find the minimum, we should go at the opposite direction of gradient
# gradient is a tuple
drift = lr * gradient

z = -x**2 - y**2 # the current function value

# Number of iterition: 30
for i in range(30):

    print('x=%.2f, y=%.2f, Function Value %.2f, Gradient(∂z/∂x, ∂z/∂y)=(%.2f, %.2f), Step(⊿x,⊿y) = (%.2f, %.2f)'
          % (x, y, z, gradient[0], gradient[1], drift[0], drift[1]))

    # vector + vector = new vector, the two parts of the new vector are assigned to x and y
    x, y = numpy.array((x, y)) + drift # Move a step, and get the new location

    z = -x ** 2 - y ** 2 # get the function value of new location

    # Get the gradient vector of new location, the input is a Tuple
    gradient = numpy.array( (-2.0 * x, -2.0 * y) )

    # Calculate the step to move, (⊿x, ⊿y)
    drift = lr * gradient
