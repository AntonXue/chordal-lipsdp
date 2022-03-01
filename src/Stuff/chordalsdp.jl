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
  use_dual :: Bool = false
  verbose :: Bool = false
end

# Set up the model for the solver call
function setup!(model, inst :: QueryInstance, opts :: ChordalSdpOptions)
  setup_start_time = time()
  vars = Dict()

  # Set up the variable where γ = [γt; γlip]
  γdim = γlength(opts.τ, inst.ffnet)
  γ = @variable(model, [1:γdim])
  vars[:γ] = γ
  @constraint(model, γ[1:γdim] .>= 0)

  # The Z matrix as a sum
  Z = makeZ(γ, opts.τ, inst.ffnet)

  # All the Zs
  cinfos = makeCliqueInfos(opts.τ, inst.ffnet)
  Zs = Vector{Any}()
  for (k, _, Ckdim) in cinfos
    # Set up the LMI for each Zk
    Zk = @variable(model, [1:Ckdim, 1:Ckdim], Symmetric)
    vars[Symbol("Z" * string(k))] = Zk
    @constraint(model, -Zk in PSDCone())
    push!(Zs, Zk)
  end

  # Assert the equality constraint and objective
  Zdim = sum(inst.ffnet.edims)
  Zksum = sum(Ec(kstart, Ckdim, Zdim)' * Zs[k] * Ec(kstart, Ckdim, Zdim) for (k, kstart, Ckdim) in cinfos)
  @constraint(model, Z .== Zksum)
  @objective(model, Min, γ[end]) # γ[end] is γlip

  # Return stuff
  setup_time = time() - setup_start_time
  return model, vars, setup_time
end

# Run the query and return the solution summary
function solve!(model, vars, opts :: ChordalSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values
end

# The interface to call
function runQuery(inst :: QueryInstance, opts :: ChordalSdpOptions)
  total_start_time = time()

  # Model setup
  if opts.use_dual
    model = Model(dual_optimizer(Mosek.Optimizer))
    set_optimizer_attribute(model, "QUIET", true)
    set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", opts.max_solve_time)
    set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", opts.solver_tol)
    set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", opts.solver_tol)
    set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", opts.solver_tol)
    if opts.verbose; @printf("\tchordalsdp model using dual\n") end

  # Model setup as primal
  else
    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "QUIET", true)
    set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", opts.max_solve_time)
    set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", opts.solver_tol)
    set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", opts.solver_tol)
    set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", opts.solver_tol)
    if opts.verbose; @printf("\tchordalsdp model using primal\n") end
  end

  # Setup and solve
  _, vars, setup_time = setup!(model, inst, opts)
  summary, values = solve!(model, vars, opts)
  total_time = time() - total_start_time
  if opts.verbose
    @printf("\tsetup time: %.3f \tsolve time: %.3f \ttotal time: %.3f \tvalue: %.3f (%s)\n",
            setup_time, summary.solve_time, total_time,
            objective_value(model), string(summary.termination_status))
  end

  return SolutionOutput(
    objective_value = objective_value(model),
    values = values,
    summary = summary,
    termination_status = string(summary.termination_status),
    total_time = total_time,
    setup_time = setup_time,
    solve_time = summary.solve_time)
end

