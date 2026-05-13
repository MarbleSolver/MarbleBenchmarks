# hakonsen.mod	LOR2-AN-MCP-15-14-11 
# Original AMPL coding by Sven Leyffer, Aug. 2005.

# Optimal tax with two factors of production No other Frills
# Taken from GAMS Model by Miles Light, Department of Economics, 
# University of Colorado, Boulder, 1999.

set I := 1..2 ;    # commodities

param lbar  := 2;             # initial labor endowment	
param kbar  := 1, > 0;        # initial capital endowment (assume positive in model)
param c0    := 3;             # utility index
param betal := 1;             # leisure consumption
param rev   := 0.5;           # revenues generated
param sigma := 0.8;           # elasticity of substitution in consumption

param alpha{I} default 0.5;   # labor intensity in production of Y
                              # try changing the alphas to 0.8 and 0.2
param phi{I} default 1;       # productivity parameter	
param beta{I} default 1;      # consumption shares
  
var Y{I} >= 0, := 1;                  # Consumption Commodities
var C >= 0, := 1;	              # Utility Production
var G >= 0;	                      # Government Good Production
var P{I} >= 0, := 1;	              # Price Of Consumption Commodities
var PC >= 0, := 1;	              # Price Of Utility
var PL >= 0, := 1;	              # Price Of Labor
var PK >= 0, := 1;	              # Price Of Capital
var PG >= 0, := 1;	              # Price Of Government Good
var GOVT >= 0;	                      # Government Agent
var T{I} >= 0, := 0.4;                # Endogenous Tax Rate
var MU >= 0, := 0;	              # Shadow Value On Government Constraint 
var TAU{I} >= 0.4, <= 0.6, := 0.5;    # Design variables for MPEC ;

# Maximize welfare
maximize welfare: C;

subject to 

   # Zero Profit For Activity I
   PROFIT{i in I}: PL^alpha[i] * PK^(1-alpha[i])  >=  phi[i] * P[i]
                   complements  Y[i] >= 0;

   # Zero Profit For Welfare Activity
   PROFITC: (betal / c0 * PL^(1-sigma) 
             + sum{i in I} beta[i]/c0*(P[i]*(1+T[i]))^(1-sigma))^(1/(1-sigma)) 
            >= PC  complements  C >= 0;
   
   # Zero Profit For Welfare Government Good
   PROFITG: PG >= PL  complements  G >= 0;

   # Market Clearance For Govt Good
   MARKETG: G * PG >= GOVT  complements  PG >= 0;

   # Market Clearance For Commodity I
   MARKET{i in I}: Y[i] * phi[i] >= (PC/(P[i]*(1+T[i])))^sigma * beta[i] * C
                   complements  P[i] >= 0;

   # Market Clearance For Labor
   MARKETL: PL * lbar >= GOVT + sum{i in I}Y[i] * P[i] * phi[i] * alpha[i] 
      	                 + PL * (PC/PL)^sigma * betal * C
            complements  PL >= 0;

   # Market Clearance For Capital
   MARKETK: PK * kbar >= sum{i in I} Y[i] * P[i] * phi[i] * (1 - alpha[i])
            complements  PK >= 0;

   # Government Income Balance
   REVENUE: GOVT >= sum{i in I} Y[i] * phi[i] * P[i] * T[i] 
            complements GOVT >= 0;

   # Household Income Balance
   INCOME: PC * C * c0 >= PL * lbar + PK * kbar  complements  PC >= 0;

   # Revenue Constraint
   REV_CONST: GOVT >= PL * rev  complements  MU >= 0;

   # Tax Definition  
   TAX{i in I}: T[i] >= MU * TAU[i]  complements  T[i] >= 0;

   # numeraire
   fix PL := 1;
