# Skipped Problems — Mixed / Non-Standard Complementarity

Problems skipped because they contain complementarity constraints that are **not** of the
required form `0 <= f(x)  complements  g(x) >= 0` (both sides explicitly bounded below by 0).

---

## bard2m.mod

**Violating constraints:**

```
c11: 0 <= -(0.4*y11 + 0.7*y12 - x11)   complements   m_c11 <= 0;
c12: 0 <= -(0.6*y11 + 0.3*y12 - x12)   complements   m_c12 <= 0;
```
Right-hand side is `m_c11 <= 0` (upper-bounded at 0), not `>= 0`.

```
d_y11: 0 = 2*(y11-4) - m_c11*0.4 - m_c12*0.6   complements   y11;
d_y12: 0 = 2*(y12-13) - m_c11*0.7 - m_c12*0.3  complements   y12;
d_y21: 0 = 2*(y21-35) - m_c21*0.4 - m_c22*0.6  complements   y21;
d_y22: 0 = 2*(y22-2)  - m_c21*0.7 - m_c22*0.3  complements   y22;
```
Left-hand side is an equality (`0 = expr`), not `0 <= expr`. Right-hand side variable
(`y11`, etc.) has no explicit `>= 0` in the complementarity expression.

---

## bilevel1m.mod

**Violating constraints:**

```
m1: -10 <= y[1] <= 20   complements   l[1];
m2: -10 <= y[2] <= 20   complements   l[2];
```
Left-hand side is a double-bounded interval (`lb <= var <= ub`) — mixed complementarity.
Right-hand side variable `l[1]`, `l[2]` is declared free (no bounds).

---

## bilevel2m.mod

**Violating constraints:**

```
m1:  0 <= y[1] <= 20   complements   l[3];
m2:  0 <= y[2] <= 20   complements   l[4];
m3:  0 <= y[3] <= 40   complements   l[7];
m4:  0 <= y[4] <= 40   complements   l[8];
```
Left-hand side is a double-bounded interval (`0 <= var <= ub`) — mixed complementarity.
Right-hand side multiplier variables `l[3]`, `l[4]`, `l[7]`, `l[8]` are declared free.

---

## TrafficSignalCycle.mod

**Violating constraints:**

```
c41 {i in S}: L[i,1] >= L0[i] + b1[i]*y[2] + b3[i]  complements  L[i,1] >= b5[i];
c4  {i in S, j in 1..N}: L[i,2*j] >= L[i,2*j-1] + b1[i]*y[2] + b3[i]  complements  L[i,2*j] >= b5[i];
c7  {i in S, j in 1..N}: L[i,2*j+1] >= L[i,2*j] + b2[i]*y[1] + b4[i]  complements  L[i,2*j+1] >= b6[i];
```
Both sides are of the form `expr >= lb` with `lb != 0`, and neither side is written as
`0 <= expr`. This is a piecewise / max-form complementarity, not the standard NCP form.
