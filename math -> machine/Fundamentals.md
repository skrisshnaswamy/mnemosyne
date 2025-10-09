# Linear Algebra

Reference: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

---
# Matrix
Think of matrix as just a way of representation. Like when we've a point in space which can be represented by a single scalar value then such a space is called a one-dimensional space and then value is just a point on that $1D$ space, and if the value is represented with 2 features or 2 values $x, y$ then the space is a $2D$ space (plane) and the $x,y,z$ point is a point in $3D$ plane / space and so on. When we've a $N$ dimensional space / hyperplane, points in such a space can be represented as **Vectors** and a collection of $M$ such vectors is neatly represented into a **Matrix**.
There is this really intriging way, where the points and space stops being all about the physical dimensions in which object exist and starts to become this virtual non-tangible "space" which starts to mean a system and "points" start to mean "samples" and each "dimension" of such a space start to mean "feature".
This basically gives Matrix 2 different perspectives to be looked from.
1. **As a container for numbers:** A neat, grid-like representation of data.
2. **As a collection of points in space:** Where each row (or column) is a vector representing a sample in a high-dimensional "feature space."
	1. This second perspective is the gateway to truly understanding data science. We stop thinking about physical 3D space and start thinking about an abstract "problem space" where 'closeness' might mean similarity between customers, pictures, or sentences.
Now there is a third perspective to Matrix.
3. **Matrices as transformation** - This perspective stops looking a matrices as just static containers of neatly represented numbers / data points; and starts to look at them as functions / machines.
	1. Machines that are capable of transforming the whole or parts of the space, it's interacting with. This is the essence of [[Fundamentals#Linear Algebra|linear algebra]] infact. ![[matrix_transformation.excalidraw]]
	2. When you multiply a vector by a matrix, you're not just doing a bunch of arithmetic. You are feeding the vector into the matrix "machine," and it spits out a new vector that has been **moved, stretched, squished, or rotated.**
	3. Imagine the entire 2D plane as a sheet of graph paper. A matrix can be a command that says "rotate the entire sheet 90 degrees," or "stretch the whole sheet so it's twice as wide." Every single point (vector) on the paper gets transformed according to the same rule, which is encoded in the numbers of the matrix.

This perspective is the key to understanding why linear algebra is so fundamental to machine learning:
- **Computer Graphics:** How do you rotate a 3D model? You multiply all of its vertices (vectors) by a "rotation matrix."
- **Dimensionality Reduction (PCA):** How do you find the most important "features" in your data? You find a matrix that squishes the "feature space" down, collapsing the least important dimensions.
- **Neural Networks:** A neural network layer is essentially a matrix multiplication (a linear transformation) followed by a non-linear activation function. Each layer warps and transforms the input data's feature space to make it easier to classify.

---
# Vectors
A vector is an array of numbers - or simply put a single column [[Fundamentals#Matrix|matrix]].
In machine learning, vectors are used to represent data points, features, and weights. *Example:* A single image pixel color can be a 3-dimensional vector (R, G, B).

---
# Function
In Math, a function is a deterministic machine that when you give a certain input, it gives a determinstic output. 
Example: $f(x) = x^2$
Here how much we vary $x$, determines, what value is outputted by $f(x)$.
And the output $f(x)$ doesn't change for the **same** value of $x$.
It's a predictable relationship. And because it's a predictable relationship, we can ask, "How much will $f(x)$ vary as we vary $x$?". And this question is what is answered by it's [[Derivative]].

This is different from a [[Random variable]]

Also, what if the function isn't as straightforward as $f(x)$? What is this machine is super complex and depends on many different inputs? A simple example would be $f(x,y)$ which depends on 2 inputs. Such a function is called a [[#Multi-variate function|multi-variate function]]. And a real life system or machine typically is a multi-variate function depending on many different inputs. And this is also how a machine-learning model (which FYI is nothing but a giant function $model(x_1, x_2, ... x_n) = Y$ is a multi-variate function)

---
# Multi-variate function
A function which takes in more than 1 input.
Example: $F(x_1, x_2, ... x_n) = Y$

---

