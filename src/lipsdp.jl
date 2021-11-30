# Implementation of LipSdp
module LipSdp

using ..Header
using ..Common
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

# Do the simplest setup so we can have the fewest opportunities for error
function M1(k :: Int, Λkdim ::Int, Λ, A, B, inst :: QueryInstance, opts :: QueryOptions)
  ffnet = inst.net
  if !(ffnet.nettype isa ReluNetwork || ffnet.nettype isa TanhNetwork)
    error("unsupported network: " * string(ffnet))
  end

  # We get different Tk depending on the sparsity pattern that we have specified
  if opts.TkSparsity isa TkαBanded
    Tk = Tkbanded(Λkdim, Λ, α=opts.TkSparsity.α)
  else
    error("unsupported TkSparsity: " * string(opts.TkSparsity))
  end

  a = 0
  b = 1
  Fkβ = Ec(k, opts.β, ffnet.λdims)

  _R11 = -2 * a * b * A' * Fkβ' * Tk * Fkβ * A
  _R12 = (a + b) * A' * Fkβ' * Tk * Fkβ * B
  _R21 = (a + b) * B' * Fkβ' * Tk * Fkβ * A
  _R22 = -2 * B' * Fkβ' * Tk * Fkβ * B
  M1 = _R11 + _R12 + _R21 + _R22
  return M1 
end

function M2(ρ, inst :: QueryInstance, opts :: QueryOptions)
  ffnet = inst.net
  E1 = E(1, ffnet.mdims)
  EK = E(ffnet.K, ffnet.mdims)
  WK = ffnet.Ms[ffnet.K][1:end, 1:end-1]
  _R1 = -ρ * E1' * E1
  _R2 = EK' * (WK' * WK) * EK
  M2 = _R1 + _R2
  return M2
end

function setup(inst :: QueryInstance, opts :: QueryOptions)
  setup_start_time = time()

  @assert inst.net isa FeedForwardNetwork
  ffnet = inst.net
  mdims = ffnet.mdims
  λdims = ffnet.λdims
  K = ffnet.K
  L = K - 1
  β = opts.β
  p = L - β

  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

  A = sum(E(j, λdims)' * ffnet.Ms[j][1:end, 1:end-1] * E(j, mdims) for j in 1:L)
  B = sum(E(j, λdims)' * E(j+1, mdims) for j in 1:L)

  M1s = Vector{Any}()
  for k in 1:p
    Λkdim = sum(λdims[k:k+β])
    Λk = @variable(model, [1:Λkdim, 1:Λkdim], Symmetric)
    @constraint(model, Λk[1:Λkdim, 1:Λkdim] .>= 0)

    M1k = M1(k, Λkdim, Λk, A, B, inst, opts)
    push!(M1s, M1k)
  end

  ρ = @variable(model)
  @constraint(model, ρ >= 0)

  _M2 = M2(ρ, inst, opts)
  M = sum(M1s) + _M2

  @SDconstraint(model, M <= 0)

  @objective(model, Min, ρ)

  setup_time = time() - setup_start_time

  return model, setup_time
end

# Run the query and return the solution summary
function solve!(model, opts :: QueryOptions)
  optimize!(model)
  return solution_summary(model)
end

# The interface to call
function run(inst :: QueryInstance, opts :: QueryOptions)
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


export M1, M2
export setup, solve!, run

end # End module

