# gnash1m.mod	QQR2-MN-10-8
# Original AMPL coding by Sven Leyffer, University of Dundee

# Formulation with mixed complementarity of ...
#
# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 8
#
# Arises from Gournot Nash equilibrium , 10 instances are available 
# (see gnash1i.dat, i=0,1,...,9)

# Number of variables:   10
# Number of constraints:  8

# ... parameters for each firm/company
param c{1..5};			# c_i
param K{1..5};			# K_i
param b{1..5};			# \beta_i

# ... parameters for each problem instance
param L;			# L
param g;			# \gamma

# ... computed constants
param gg := 5000^(1/g);

var x >= 0, <= L;
var y{1..4};
var l{1..4};   		        # Multipliers
var Q = x+y[1]+y[2]+y[3]+y[4];	# defined variable Q

minimize f: c[1]*x + b[1]/(b[1]+1)*K[1]^(-1/b[1])*x^((1+b[1])/b[1])
		- x*( gg*Q^(-1/g) );

subject to 

   F1: 0 = ( c[2] + K[2]^(-1/b[2])*y[1] ) - ( gg*Q^(-1/g) ) 
				- y[1]*( -1/g*gg*Q^(-1-1/g) ) - l[1];
   F2: 0 = ( c[3] + K[3]^(-1/b[3])*y[2] ) - ( gg*Q^(-1/g) ) 
				- y[2]*( -1/g*gg*Q^(-1-1/g) ) - l[2];
   F3: 0 = ( c[4] + K[4]^(-1/b[4])*y[3] ) - ( gg*Q^(-1/g) ) 
				- y[3]*( -1/g*gg*Q^(-1-1/g) ) - l[3];
   F4: 0 = ( c[5] + K[5]^(-1/b[5])*y[4] ) - ( gg*Q^(-1/g) ) 
				- y[4]*( -1/g*gg*Q^(-1-1/g) ) - l[4];

   g1: 0 <= y[1] <= L  complements   l[1];
   g3: 0 <= y[2] <= L  complements   l[2];
   g5: 0 <= y[3] <= L  complements   l[3];
   g7: 0 <= y[4] <= L  complements   l[4];



        

