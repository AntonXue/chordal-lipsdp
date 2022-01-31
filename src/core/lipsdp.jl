# Implementation of LipSdp
module LipSdp

using ..Header
using ..Common
using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Mosek
using Printf

# Options
@with_kw struct LipSdpOptions
  τ :: Int = 0
  max_solve_time :: Float64 = 30.0  # Timeout in seconds
  solver_tol :: Float64 = 1e-6
  verbose :: Bool = false
end

# Set up the model for the solver call
function setup!(model, inst :: QueryInstance, opts :: LipSdpOptions)
  setup_start_time = time()
  vars = Dict()

  # Set up M1
  Tdim = sum(inst.ffnet.fdims)
  γdim = γlength(Tdim, opts.τ)
  γ = @variable(model, [1:γdim])
  vars[:γ] = γ
  @constraint(model, γ[1:γdim] .>= 0)
  T = makeT(Tdim, γ, opts.τ)
  A, B = makeA(inst.ffnet), makeB(inst.ffnet)
  M1 = makeM1(T, A, B, inst.ffnet)

  # Set up M2
  ρ = @variable(model)
  vars[:ρ] = ρ
  @constraint(model, ρ >= 0)
  M2 = makeM2(ρ, inst.ffnet)

  # Impose the LMI constraint and objective
  Z = M1 + M2
  @SDconstraint(model, Z <= 0)
  @objective(model, Min, ρ)

  # Return information
  setup_time = time() - setup_start_time
  if opts.verbose; @printf("\tsetup time: %.3f\n", setup_time) end
  return model, vars, setup_time
end

# Run the query and return the solution summary
function solve!(model, vars, opts :: LipSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  if opts.verbose; @printf("\tsolve time: %.3f\n", summary.solve_time) end
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values
end

# The interface to call
function run(inst :: QueryInstance, opts :: LipSdpOptions)
  total_start_time = time()

  # Model setup
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "MSK_DPAR_OPTIMIZER_MAX_TIME" => opts.max_solve_time,
    "INTPNT_CO_TOL_REL_GAP" => opts.solver_tol,
    "INTPNT_CO_TOL_PFEAS" => opts.solver_tol,
    "INTPNT_CO_TOL_DFEAS" => opts.solver_tol))

  # Setup and solve
  _, vars, setup_time = setup!(model, inst, opts)
  summary, values = solve!(model, vars, opts)

  # Return
  total_time = time() - total_start_time
  if opts.verbose; @printf("\ttotal time: %.3f\n", total_time) end
  return SolutionOutput(
    objective_value = objective_value(model),
    values = values,
    summary = summary,
    termination_status = string(summary.termination_status),
    total_time = total_time,
    setup_time = setup_time,
    solve_time = summary.solve_time)
end

#
export LipSdpOptions, WholeTSetup, SummedXSetup
export setup!, solve!, run

end # End module

