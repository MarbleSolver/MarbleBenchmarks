# pack-comp1.mod LQR2-MN-v-v
# Original AMPL coding by Sven Leyffer, University of Dundee
#
# An MPEC from Outrata, Kocvara & Zowe, Nonsmooth Approach to
# Optimization Problems with Equilibrium Constraints, Kluwer, 1998.
#
# A packaging problem (see Section 9.2, Example 9.3):
# Minimize membrane surface under the condition that
# the membrane comes into contact with a compliant obstacle.
#
# A 2D obstacle problem on [0,1] x [0,1].
#
# Several discretizations are available (see pack-comp-i.dat,
# i=8,16,32).
#
# Formulation uses FE discretization with similar data
# structure as provided by triangulization routines.

using JuMP

function build_model(n::Int,
                     bnd_nodes::Set{Int},
                     fix_nodes::Set{Int},
                     Omega0::Set{Int},
                     elements::Set{Tuple{Int,Int,Int}},
                     i_ref::Vector{Int},
                     j_ref::Vector{Int},
                     u0::Dict{Int,Float64})
    h = 1/n
    Nnd = (n+1)*(n+1)
    nodes = 1:Nnd
    int_nodes = setdiff(Set(nodes), bnd_nodes)
    c = 2.0

    # ... node positions (y fixed)
    y_param = [j_ref[i]*h for i in nodes]

    model = Model()

    # ... parameterization of "moving" boundary (alpha)
    @variable(model, 0.6 <= a[0:n] <= 1.0)
    for i in 0:n
        set_start_value(a[i], 1.0)
    end

    # ... node positions (x depends on a - moving boundary)
    x_expr(i) = i in fix_nodes ? i_ref[i]*h :
                0.5 + (i_ref[i] - n/2)*(2*a[j_ref[i]] - 1)/n

    # ... determinant of Je transformation for element stiffness matrix
    detJe_expr(i, j, k) = (x_expr(j) - x_expr(i))*(y_param[k] - y_param[i]) -
                           (y_param[j] - y_param[i])*(x_expr(k) - x_expr(i))

    # ... obstacle
    xi_expr(i) = -0.04*(x_expr(i)^2 + (y_param[i]^2 - 0.25)^2)

    # ... unknown height of membrane
    @variable(model, u[nodes])

    # ... slack variables to deal with complementarity
    @variable(model, s1[i in int_nodes] >= 0.0)

    # ... global load vector (load is f(x,y) = -1.0 constant)
    l_expr(i) = (- sum(detJe_expr(ii, j, k) / 6.0 for (ii, j, k) in elements if ii == i; init=0.0)
                 - sum(detJe_expr(j, ii, k) / 6.0 for (j, ii, k) in elements if ii == i; init=0.0)
                 - sum(detJe_expr(k, j, ii) / 6.0 for (k, j, ii) in elements if ii == i; init=0.0))

    # ... set up PDE equation at all internal nodes here
    Au_expr(i) = (
        sum((( (y_param[k]-y_param[ii])^2 + (x_expr(k)-x_expr(ii))^2 + (x_expr(j)-x_expr(ii))^2
               - 2*(x_expr(k)-x_expr(ii))*(x_expr(j)-x_expr(ii))                              ) * u[ii]
             + (-(y_param[k]-y_param[ii])^2 - (x_expr(k)-x_expr(ii))^2 + (x_expr(k)-x_expr(ii))*(x_expr(j)-x_expr(ii)) ) * u[j]
             + (-(x_expr(j)-x_expr(ii))^2 + (x_expr(k)-x_expr(ii))*(x_expr(j)-x_expr(ii))                 ) * u[k]
            ) / ( 2*detJe_expr(ii, j, k) )
           for (ii, j, k) in elements if ii == i; init=0.0)

      + sum((( (y_param[k]-y_param[j])^2 + (x_expr(k)-x_expr(j))^2                           ) * u[i]
             + (-(y_param[k]-y_param[j])^2 - (x_expr(k)-x_expr(j))^2 + (x_expr(k)-x_expr(j))*(x_expr(i)-x_expr(j)) ) * u[j]
             + (-(x_expr(k)-x_expr(j))*(x_expr(i)-x_expr(j))                                 ) * u[k]
            ) / ( 2*detJe_expr(j, i, k) )
           for (j, ii, k) in elements if ii == i; init=0.0)

      + sum((( (x_expr(j)-x_expr(k))^2                            ) * u[i]
             + (-(x_expr(i)-x_expr(k))*(x_expr(j)-x_expr(k))                  ) * u[j]
             + (-(x_expr(j)-x_expr(k))^2 + (x_expr(i)-x_expr(k))*(x_expr(j)-x_expr(k))  ) * u[k]
            ) / ( 2*detJe_expr(k, j, i) )
           for (k, j, ii) in elements if ii == i; init=0.0)
    )

    @objective(model, Min, h/2 * sum(a[i] + a[i-1] for i in 1:n))

    # ... boundary conditions
    @constraint(model, bnd_cond[i in bnd_nodes], u[i] == u0[i])

    # ... region where membrane is fixed to obstacle
    @constraint(model, fix_mem[i in Omega0], u[i] - xi_expr(i) - c*(l_expr(i) - Au_expr(i)) <= 0)

    # ... constraint on slope of "moving" boundary
    @constraint(model, slope[i in 1:n], -3*h <= a[i-1] - a[i] <= 3*h)

    # ... FE approx to Laplacian
    @constraint(model, PDE[i in int_nodes], s1[i] == Au_expr(i) - l_expr(i))

    # ... obstacle lower bound
    @variable(model, obst_s[i in int_nodes] >= 0)
    @constraint(model, obst_def[i in int_nodes], obst_s[i] == u[i] - xi_expr(i) - c*(l_expr(i) - Au_expr(i)))
    @constraint(model, obst[i in int_nodes], [s1[i], obst_s[i]] in SOS1())

    return model
end
