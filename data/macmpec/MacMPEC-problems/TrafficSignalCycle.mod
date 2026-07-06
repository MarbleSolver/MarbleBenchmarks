#################################################################################################
#  March/2012											#
#												#
#  This problem is described in "Optimal cycle for a signalized intersection using Global       #
#  Optimization and Complementarity, by Isabel M. Ribeiro and M. Lurdes Simoes in Sociedad de   #    
#  Estadistica e Investigacion Operativa 2010, DOI 10.1007/s11750-010-0167-3                    #
#												#	
#  Coded by Teofilo Melo, Teresa Monteiro and Joao Matias                                       #   
#################################################################################################					

# SETS
				
set S; 					# traffic streams
set K; 					# time instants 

# PARAMETERS

param N;				# time periods  
param xmax; 				# maximum queue length in each traffic stream
param gmax; 				# maximum duration for green and red time
param gmin; 				# minimum duration for green and red time
param lambda {i in S}; 			# average arrival rate of vehicles in traffic each stream 
param L0 {i in S};		 	# queue length in traffic stream S at initial time t0
param b1 {i in S};			# vectors b1, b2, b3, b4, b5 and b6  
param b2 {i in S};
param b3 {i in S};
param b4 {i in S};
param b5 {i in S};
param b6 {i in S};

# VARIABLES 

var L {i in S,j in K}; 			# queue length in traffic stream S at instant time K
var y {1..2};				# time duration for red and green time 

# FUNCTION TO MINIMIZE: average waiting time over all queues

  minimize Obj: sum {i in S} 1/lambda[i]*((1/(2*(2*N+1))*L0[i]+ sum {j in 1..2*N} (1/(2*N+1)* L[i,j]) + 1/(2*(2*N+1))*L[i,2*N+1]));

# CONSTRAINTS 

subject to 

 c1 {i in S, j in K }: 0 <= L[i,j]<=xmax;

 c21 {i in S} : L[i,1]>=L0[i] + b1[i]*y[2]+b3[i];

 c31 {i in S} : L[i,1]>=b5[i];

 c41 {i in S} : L[i,1]>=L0[i] + b1[i]*y[2]+b3[i]complements L[i,1]>=b5[i];
 
 c2 {i in S, j in 1..N} : L[i, 2*j]>=L[i,2*j-1] + b1[i]*y[2]+b3[i];

 c3 {i in S, j in 1..N} : L[i, 2*j]>=b5[i];

 c4 {i in S, j in 1..N} : L[i, 2*j]>=L[i,2*j-1] + b1[i]*y[2]+b3[i] complements L[i, 2*j]>=b5[i];

 c5 {i in S, j in 1..N} : L[i, 2*j+1]>=L[i,2*j] + b2[i]*y[1]+b4[i];

 c6 {i in S, j in 1..N} : L[i, 2*j+1]>=b6[i];

 c7 {i in S, j in 1..N}: L[i, 2*j+1]>=L[i,2*j] + b2[i]*y[1]+b4[i] complements L[i, 2*j+1]>=b6[i];

 c8 {i in 1..2}: gmin<=y[i]<=gmax;

