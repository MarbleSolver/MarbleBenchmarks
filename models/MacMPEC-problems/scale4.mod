# scale4.mod
#

param a default 100;

var x1;
var x2;

minimize f: (a*x1 - 1)^2 + (a*x2 - 1)^2;

subject to 

   compl: 0 <= x1  complements  x2 >= 0;


