# Implmenetation of a split version of LipSdp
module ChordalSdp

using ..Header
using ..Common
using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Dualization
using Printf

# How the construction is done
@with_kw struct ChordalSdpOptions
  τ :: Int = 0
  max_solve_time :: Float64 = 30.0
  solver_tol :: Float64 = 1e-6
  use_dual :: Bool = true
  verbose :: Bool = false
end

# Set up the model for the solver call
function setup!(model, inst :: QueryInstance, opts :: ChordalSdpOptions)
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

  # All the Zs
  cinfos = makeCliqueInfos(opts.τ, inst.ffnet)
  Zs = Vector{Any}()
  for (k, _, Ckdim) in cinfos
    Zk = @variable(model, [1:Ckdim, 1:Ckdim], Symmetric)
    vars[Symbol("Z" * string(k))] = Zk
    @SDconstraint(model, Zk <= 0)
    push!(Zs, Zk)
  end

  # Assert the equality constraint and objective
  Zdim = sum(inst.ffnet.edims)
  Zksum = sum(Ec(kstart, Ckdim, Zdim)' * Zs[k] * Ec(kstart, Ckdim, Zdim) for (k, kstart, Ckdim) in cinfos)
  @constraint(model, M1 + M2 .== Zksum)
  @objective(model, Min, ρ)

  if opts.verbose
    for (fst, snd) in list_of_constraint_types(model)
      @printf("\tfst constr: %s\n", fst)
      @printf("\tsnd constr: %s\n", snd)
      num_of_cons = num_constraints(model, fst, snd) 
      @printf("\tnum constrs: %d\n", num_of_cons)
      @printf("\n")
    end
  end

  # Return stuff
  setup_time = time() - setup_start_time
  if opts.verbose; @printf("\tsetup time: %.3f\n", setup_time) end
  return model, vars, setup_time
end

# Run the query and return the solution summary
function solve!(model, vars, opts :: ChordalSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  if opts.verbose; @printf("\tsolve time: %.3f\n", summary.solve_time) end
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values
end

# The interface to call
function run(inst :: QueryInstance, opts :: ChordalSdpOptions)
  total_start_time = time()

  # Model setup
  if opts.use_dual
    model = Model(dual_optimizer(optimizer_with_attributes(
      Mosek.Optimizer,
      "QUIET" => true,
      "MSK_DPAR_OPTIMIZER_MAX_TIME" => opts.max_solve_time,
      "INTPNT_CO_TOL_REL_GAP" => opts.solver_tol,
      "INTPNT_CO_TOL_PFEAS" => opts.solver_tol,
      "INTPNT_CO_TOL_DFEAS" => opts.solver_tol)))
    if opts.verbose; @printf("\tusing dualization\n") end
  else
    model = Model(optimizer_with_attributes(
      Mosek.Optimizer,
      "QUIET" => true,
      "MSK_DPAR_OPTIMIZER_MAX_TIME" => opts.max_solve_time,
      "INTPNT_CO_TOL_REL_GAP" => opts.solver_tol,
      "INTPNT_CO_TOL_PFEAS" => opts.solver_tol,
      "INTPNT_CO_TOL_DFEAS" => opts.solver_tol))
    if opts.verbose; @printf("\tNOT using dualization\n") end
  end

  # Setup and solve
  _, vars, setup_time = setup!(model, inst, opts)
  summary, values = solve!(model, vars, opts)

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

export ChordalSdpOptions
export run

end # End module

