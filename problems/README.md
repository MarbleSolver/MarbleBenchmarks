# Useful QPCC Reformulations of Functions

## Absolute value of a vector `v`:

$|v|$ can be written as the following LP:

$$
\begin{align*}
\arg\min_s \quad &\bm{1}^T s \\ 
\textrm{subject to} \quad & s \geq v \\
& s \geq -v
\end{align*}
$$

with the following KKT conditions:

$$
\begin{align*}
L(s, \lambda_1, \lambda_2) &= \bm{1}^T s - \lambda_1^T(s - v) - \lambda_2^T(s + v) \\
\bm{1} - \lambda_1 - \lambda_2 &= 0 \Longleftrightarrow \lambda_2 = \bm{1} - \lambda_1 \\
0 \leq \lambda_1 \perp (s - v) &\geq 0 \\
0 \leq (\bm{1} - \lambda_1) \perp (s + v) &\geq 0 \\
\end{align*}
$$

## Min over a vector `v`:

The min over a vector $v$ can be written as the following LP:

$$
\begin{align*}
\min_s \quad & -s \\
\textrm{subject to} \quad & v \geq \bm{1}s
\end{align*}
$$

with the following KKT conditions:

$$
\begin{align*}
L(s, \lambda) &= -s - \lambda^T(v - \bm{1}s) \\
1 - \bm{1}^T \lambda &= 0 \\
0 \leq \lambda \perp (v - \bm{1}s) &\geq 0
\end{align*}
$$

## Max over a vector `v`:

The max over a vector $v$ can be written as the following LP:

$$
\begin{align*}
\min_s \quad & s \\
\textrm{subject to} \quad & \bm{1}s \geq v
\end{align*}
$$

with the following KKT conditions:

$$
\begin{align*}
L(s, \lambda) &= s - \lambda^T(\bm{1}s - v) \\
1 - \bm{1}^T \lambda &= 0 \\
0 \leq \lambda \perp (\bm{1}s - v) &\geq 0
\end{align*}
$$

## Pairwise min between two vectors `v` and `w`:

The pairwise min between two vectors $v,w \in \mathbb{R}^n$ can be written as the following LP:

$$
\begin{align*}
\arg\min_s \quad & -\bm{1}^T s \\
\textrm{subject to} \quad & v \geq s \\
& w \geq s
\end{align*}
$$

with the following KKT conditions:

$$
\begin{align*}
L(s,\lambda_1,\lambda_2)
&=
-\bm{1}^T s
-
\lambda_1^T(v-s)
-
\lambda_2^T(w-s)
\\
-\bm{1}+\lambda_1+\lambda_2 &= 0
\Longleftrightarrow
\lambda_2 = \bm{1}-\lambda_1
\\
0 \leq \lambda_1 \perp (v-s) &\geq 0
\\
0 \leq (\bm{1}-\lambda_1) \perp (w-s) &\geq 0.
\end{align*}
$$

## Pairwise max between two vectors `v` and `w`:

The pairwise max between two vectors $v,w \in \mathbb{R}^n$ can be written as the following LP:

$$
\begin{align*}
\arg\min_s \quad & \bm{1}^T s \\
\textrm{subject to} \quad & s \geq v \\
& s \geq w
\end{align*}
$$

with the following KKT conditions:

$$
\begin{align*}
L(s,\lambda_1,\lambda_2)
&=
\bm{1}^T s
-
\lambda_1^T(s-v)
-
\lambda_2^T(s-w)
\\
\bm{1}-\lambda_1-\lambda_2 &= 0
\Longleftrightarrow
\lambda_2 = \bm{1}-\lambda_1
\\
0 \leq \lambda_1 \perp (s-v) &\geq 0
\\
0 \leq (\bm{1}-\lambda_1) \perp (s-w) &\geq 0.
\end{align*}
$$