# sl1.mod	QQR2-MN-11-6
# Original AMPL coding by Sven Leyffer

# A QPEC obtained by varying the rhs of HS21 (a QP)
# from an idea communicated by S. Scholtes

set I := 1..3;

param zl{I};
param zu{I} default Infinity;

var x{1..2};
var z{i in I} >= zl[i], <= zu[i];

var l{I} >= 0;

minimize f:
    (x[1] - 2)^2 + x[2]^2;

subject to

   KKT1: 0.02*x[1] - 10*l[1] - l[2] = 0;

   KKT2: 2*x[2] - l[1] - l[3] = 0;

   lin_1: 0 <= 10*x[1] + x[2] - (10 + z[1]) complements l[1] >= 0;

   lin_2: 0 <= x[1] - (2 + z[2]) complements l[2] >= 0;

   lin_3: 0 <= x[2] - 50*z[3] complements l[3] >= 0;

data;

param zl :=
    1  10
    2  0.01
    3  0
;

param zu :=
    2  10
    3  1
;