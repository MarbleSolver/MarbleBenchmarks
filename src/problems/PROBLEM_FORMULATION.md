# Useful QPCC Reformulations of Functions

## Absolute value of a scalar or vector `v`:

For a scalar $v$, the absolute value $s = |v|$ can be represented as the solution of the LP:

$$
\begin{align*}
\operatorname*{arg\,min}_{s} \quad & s \\
\text{subject to} \quad & s \geq v \\
& s \geq -v.
\end{align*}
$$

with the following KKT conditions:

$$
\begin{align*}
L(s, \lambda_+, \lambda_-) &= s - \lambda_+(s - v) - \lambda_-(s + v) \\
1 - \lambda_+ - \lambda_- &= 0 \\
0 \leq \lambda_+ \perp (s - v) &\geq 0 \\
0 \leq \lambda_- \perp (s + v) &\geq 0.
\end{align*}
$$

For a vector $v \in \mathbb{R}^n$, apply the same construction componentwise:

$$
\begin{align*}
\operatorname*{arg\,min}_{s} \quad & \mathbf{1}^T s \\
\text{subject to} \quad & s \geq v \\
& s \geq -v.
\end{align*}
$$

The vector KKT conditions are:

$$
\begin{align*}
L(s, \lambda_+, \lambda_-) &= \mathbf{1}^T s - \lambda_+^T(s - v) - \lambda_-^T(s + v) \\
\mathbf{1} - \lambda_+ - \lambda_- &= 0 \\
0 \leq \lambda_+ \perp (s - v) &\geq 0 \\
0 \leq \lambda_- \perp (s + v) &\geq 0.
\end{align*}
$$

## Sign of a scalar or vector `x`:

For a scalar $x$, the sign value $s = \operatorname{sign}(x)$ can be represented as the solution of the box-constrained LP:

$$
\begin{align*}
s = \operatorname*{arg\,max}_{\sigma} \quad & x \sigma \\
\text{subject to} \quad & -1 \leq \sigma \leq 1.
\end{align*}
$$

Equivalently, use the minimization form:

$$
\begin{align*}
s = \operatorname*{arg\,min}_{\sigma} \quad & -x \sigma \\
\text{subject to} \quad & 1 - \sigma \geq 0 \\
& 1 + \sigma \geq 0.
\end{align*}
$$

with the following KKT conditions:

$$
\begin{align*}
L(\sigma,\lambda_+,\lambda_-)
&=
-x\sigma
-
\lambda_+(1-\sigma)
-
\lambda_-(1+\sigma)
\\
-x + \lambda_+ - \lambda_- &= 0 \\
0 \leq \lambda_+ \perp (1-\sigma) &\geq 0 \\
0 \leq \lambda_- \perp (1+\sigma) &\geq 0.
\end{align*}
$$

For a vector $x \in \mathbb{R}^n$, apply the same construction componentwise:

$$
\begin{align*}
s = \operatorname*{arg\,max}_{\sigma} \quad & x^T \sigma \\
\text{subject to} \quad & -\mathbf{1} \leq \sigma \leq \mathbf{1}.
\end{align*}
$$

The vector KKT conditions are:

$$
\begin{align*}
L(\sigma,\lambda_+,\lambda_-)
&=
-x^T\sigma
-
\lambda_+^T(\mathbf{1}-\sigma)
-
\lambda_-^T(\mathbf{1}+\sigma)
\\
-x + \lambda_+ - \lambda_- &= 0 \\
0 \leq \lambda_+ \perp (\mathbf{1}-\sigma) &\geq 0 \\
0 \leq \lambda_- \perp (\mathbf{1}+\sigma) &\geq 0.
\end{align*}
$$

## Min over a vector `v`:

The min over a vector $v$ can be written as the following LP:

$$
\begin{align*}
\operatorname*{minimize}_{s} \quad & -s \\
\text{subject to} \quad & v \geq \mathbf{1}s
\end{align*}
$$

with the following KKT conditions:

$$
\begin{align*}
L(s, \lambda) &= -s - \lambda^T(v - \mathbf{1}s) \\
1 - \mathbf{1}^T \lambda &= 0 \\
0 \leq \lambda \perp (v - \mathbf{1}s) &\geq 0
\end{align*}
$$

## Max over a vector `v`:

The max over a vector $v$ can be written as the following LP:

$$
\begin{align*}
\operatorname*{minimize}_{s} \quad & s \\
\text{subject to} \quad & \mathbf{1}s \geq v
\end{align*}
$$

with the following KKT conditions:

$$
\begin{align*}
L(s, \lambda) &= s - \lambda^T(\mathbf{1}s - v) \\
1 - \mathbf{1}^T \lambda &= 0 \\
0 \leq \lambda \perp (\mathbf{1}s - v) &\geq 0
\end{align*}
$$

## Pairwise min between two vectors `v` and `w`:

The pairwise min between two vectors $v,w \in \mathbb{R}^n$ can be written as the following LP:

$$
\begin{align*}
\operatorname*{arg\,min}_{s} \quad & -\mathbf{1}^T s \\
\text{subject to} \quad & v \geq s \\
& w \geq s
\end{align*}
$$

with the following KKT conditions:

$$
\begin{align*}
L(s, \lambda_1, \lambda_2)
&=
-\mathbf{1}^T s
-
\lambda_1^T(v - s)
-
\lambda_2^T(w - s)
\\
-\mathbf{1} + \lambda_1 + \lambda_2 &= 0 \\
0 \leq \lambda_1 \perp (v - s) &\geq 0
\\
0 \leq \lambda_2 \perp (w - s) &\geq 0.
\end{align*}
$$

## Pairwise max between two vectors `v` and `w`:

The pairwise max between two vectors $v,w \in \mathbb{R}^n$ can be written as the following LP:

$$
\begin{align*}
\operatorname*{arg\,min}_{s} \quad & \mathbf{1}^T s \\
\text{subject to} \quad & s \geq v \\
& s \geq w
\end{align*}
$$

with the following KKT conditions:

$$
\begin{align*}
L(s, \lambda_1, \lambda_2)
&=
\mathbf{1}^T s
-
\lambda_1^T(s - v)
-
\lambda_2^T(s - w)
\\
\mathbf{1} - \lambda_1 - \lambda_2 &= 0
\\
0 \leq \lambda_1 \perp (s - v) &\geq 0
\\
0 \leq \lambda_2 \perp (s - w) &\geq 0.
\end{align*}
$$
