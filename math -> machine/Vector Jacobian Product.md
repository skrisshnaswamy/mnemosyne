---
aliases:
  - VJP
---

##### Definition
Given a vector $v∈R^m$ and a Jacobian matrix $J∈R^{m×n}$, the VJP is the product $v^{T}J$, resulting in a row vector of shape $1×n$.

##### Intuition
The VJP is the core engine of [[Backpropagation]]. Imagine you already know the gradient of your final loss $L$ with respect to a layer's output $y$ (this is your vector $v$ ). You want to find the gradient of the loss with respect to that layer's input $x$ . The VJP efficiently calculates this for you: 
it uses the upstream gradient $v$ and the local derivative $J$ to find the downstream gradient $∂L/∂x$ .

$VJP = ∇_x ( v·f(x) )$.

Simple Example: 
$$
f(x_1,x_2) = 
\begin{bmatrix}
x_1+2x_2 & 3x_1−x_2
\end{bmatrix}
$$
The jacobian (i.e. first-order derivative) is
$$
J = 
\begin{bmatrix}  
1 & 2 \\  
3 & -1  
\end{bmatrix}
$$


So for a 
$$
v = 
\begin{bmatrix}  
4 & 5
\end{bmatrix}
$$
then
$$
VJP = 
\begin{bmatrix}  
4 & 5
\end{bmatrix} . J
= 
\begin{bmatrix}  
19 & 3
\end{bmatrix}
$$
