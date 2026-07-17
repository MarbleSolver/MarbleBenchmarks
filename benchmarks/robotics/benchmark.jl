using MarbleBenchmarks

const RESULTS_DIR = joinpath(@__DIR__, "results")
const PROBLEMS = list_problems(:robotics)

## Benchmark Gurobi SOS1
GUROBI_OPTIONS = (
    TimeLimit = 600.0, # seconds
    MIPGap = 1e-4,
    FeasibilityTol = 1e-6,
)
df_gurobi = Benchmarks.run_benchmark("gurobi", Benchmarks.gurobi_solve; problems=PROBLEMS, save=true, results_dir=RESULTS_DIR, options=GUROBI_OPTIONS)

## Benchmark CRISP
CRISP_OPTIONS = (;
    # weighted_mode=1.0,
    # weighted_tol_factor=1.0,
    constraint_tol=1e-6,
)
df_crisp  = Benchmarks.run_benchmark("crisp", Benchmarks.crisp_solve; problems=PROBLEMS, save=true, results_dir=RESULTS_DIR, options=CRISP_OPTIONS)

## Benchmark LCQPow
using LCQPow

LCQPOW_OPTIONS = (
    use_dummy_constant = true,
    stationarity_tolerance = 1e-7,     # default ~2e-10 is too tight
    max_penalty_parameter = 1e10,      # default 1e8 caps out before complementarity converges
    qp_solver = LCQPow.QPOASES_SPARSE, # qpOASES sparse
)
df_lcqpow = Benchmarks.run_benchmark("lcqpow", Benchmarks.lcqpow_solve; problems=PROBLEMS, save=true, results_dir=RESULTS_DIR, options=LCQPOW_OPTIONS)

## Benchmark Marble
MARBLE_OPTIONS = (
    max_iters = 2000,
    relaxation_min = 1e-10,
    penalty_max = 1e6,
    convergence_eq_violation = 1e-6,
    convergence_ineq_violation = 1e-6,
    convergence_comp_violation = 1e-6,
    inertia_warmstart = false,
    comp_init_random = true,
    comp_init_seed = 42,
    relaxation_scaling = 0.5,
)
df_marble = Benchmarks.run_benchmark("marble", Benchmarks.marble_solve; problems=PROBLEMS, save=true, results_dir=RESULTS_DIR, options=MARBLE_OPTIONS)
