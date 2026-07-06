# Marble Benchmarks

Benchmark problems and driver code for comparing Marble against MPCC solvers like CCOpt, Gurobi, CRISP, LCQPow.

## Running Gurobi

Gurobi requires an academic license to use `Gurobi.jl`. Follow instructions on Gurobi's website for acquiring and registering a license key before running this code.

## Running CCOpt

CCOpt with HSL linear system solvers requires an academic license. Follow the instructions on [the installation page of MadNLP](https://madsuite.org/MadNLP.jl/stable/installation/) to first obtain a license and then ensure `HSL_jll.jl` is installed in the base Julia `1.x` environment. 

## Running Marble

Clone [the Marble repo](https://github.com/MarbleSolver/Marble) and follow the build instructions from the Marble README. Then, install Marble's Julia package in dev mode in the `MarbleBenchmarks` environment.

## Running CRISP

Clone [the CRISP fork](https://github.com/micahreich/CRISP) and follow the build instructions from its README to build the C++ bindings. Then install its Julia package (in the `julia/` subdirectory) in dev mode in the `MarbleBenchmarks` environment:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path="/path/to/CRISP/julia")'
```

## Running LCQPow

Clone [the LCQPow fork](https://github.com/micahreich/LCQPow) and follow the build instructions from its README to build the C++ bindings. Then install its Julia package (in the `interfaces/julia/` subdirectory) in dev mode in the `MarbleBenchmarks` environment:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path="/path/to/LCQPow/interfaces/julia")'
```

# Installing MarbleBenchmarks

Once the above steps have been completed, from the root of the `MarbleBenchmarks` repository, enter into a new Julia terminal and precompile the package:

```bash
julia --project=. -e 'using Pkg; Pkg.precompile(strict=true)'
```