# Implmenetation of a split version of LipSdp
module SplitLipSdp

using ..Header
using ..Common
using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

@with_kw struct SplitLipSdpOptions
  verbose :: Bool = false
end

function setup(inst :: QueryInstance, opts :: SplitLipSdpOptions)
  setup_start_time = time()
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

  ffnet = inst.net
  mdims = ffnet.mdims
  λdims = ffnet.λdims
  L = ffnet.L
  β = inst.β
  p = inst.p

  Ys = Vector{Any}()
  for k in 1:inst.p
    Λkdim = sum(λdims[k:k+β])
    Λk = @variable(model, [1:Λkdim, 1:Λkdim], Symmetric)
    @constraint(model, Λk[1:Λkdim, 1:Λkdim] .>= 0)
    Tk = makeT(Λkdim, Λk, inst.pattern)
    Yk = makeY(k, β, Tk, ffnet)
    push!(Ys, Yk)
  end

  ρ = @variable(model)
  @constraint(model, ρ >= 0)
  Yinit = makeYinit(β, ρ, ffnet)
  Yfinal = makeYfinal(β, ffnet)

  # Make each Zk and assert the NSD constraint
  for k in 1:p
    Zk = Ys[k]
    if k == 1; Zk += Yinit end
    if k == p; Zk += Yfinal end
    @SDconstraint(model, Zk <= 0)
  end

  # Set up objective and return
  @objective(model, Min, ρ)
  setup_time = time() - setup_start_time
  return model, setup_time
end

# Run the query and return the solution summary
function solve!(model, opts :: SplitLipSdpOptions)
  optimize!(model)
  return solution_summary(model)
end

# The interface to call
function run(inst :: QueryInstance, opts :: SplitLipSdpOptions)
  run_start_time = time()
  model, setup_time = setup(inst, opts)
  summary = solve!(model, opts)

  run_time = time() - run_start_time
  output = SolutionOutput(
            model = model,
            summary = summary,
            status = string(summary.termination_status),
            objective_value = objective_value(model),
            total_time = run_time,
            setup_time = setup_time,
            solve_time = summary.solve_time)
  return output
end

export SplitLipSdpOptions
export run

end # End module

