# Implementation of LipSdp
module LipSdp

using ..Header
using ..Common
using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

# How is the construction peformed?
abstract type LipSdpSetupMethod end
struct WholeTSetup <: LipSdpSetupMethod end
struct PartialM1Setup <: LipSdpSetupMethod end
struct PartialYSetup <: LipSdpSetupMethod end

@with_kw struct LipSdpOptions
  setupMethod :: LipSdpSetupMethod
end

# Construct M1, or smaller variants depending on what is queried with
function M1(T, A, B, ffnet :: FeedForwardNetwork)
  if !(ffnet.nettype isa ReluNetwork || ffnet.nettype isa TanhNetwork)
    error("unsupported network: " * string(ffnet))
  end

  a = 0
  b = 1
  _R11 = -2 * a * b * A' * T * A
  _R12 = (a + b) * A' * T * B
  _R21 = (a + b) * B' * T * A
  _R22 = -2 * B' * T * B
  M1 = _R11 + _R12 + _R21 + _R22
  return M1
end

# Construct M2
function M2(ρ, ffnet :: FeedForwardNetwork)
  E1 = E(1, ffnet.mdims)
  EK = E(ffnet.K, ffnet.mdims)
  WK = ffnet.Ms[ffnet.K][1:end, 1:end-1]
  _R1 = -ρ * E1' * E1
  _R2 = EK' * (WK' * WK) * EK
  M2 = _R1 + _R2
  return M2
end

# Compute all of T then query M1
function setupViaWholeT(inst :: QueryInstance, opts :: LipSdpOptions)
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
  p = L - β

  # Set up the T matrix as a sum
  Ts = Vector{Any}()
  for k in 1:p
    Λkdim = sum(λdims[k:k+β])
    Λk = @variable(model, [1:Λkdim, 1:Λkdim], Symmetric)
    @constraint(model, Λk[1:Λkdim, 1:Λkdim] .>= 0)

    if inst.innerSparsity isa TkBanded
      Tk = Tkbanded(Λkdim, Λk, α=inst.innerSparsity.α)
    else
      error("unsupported TkSparsity: " * string(inst.innerSparsity))
    end

    Fkβ = Ec(k, β, ffnet.λdims)
    push!(Ts, Fkβ' * Tk * Fkβ)
  end

  # Query M1
  T = sum(Ts)
  A = sum(E(j, λdims)' * ffnet.Ms[j][1:end, 1:end-1] * E(j, mdims) for j in 1:L)
  B = sum(E(j, λdims)' * E(j+1, mdims) for j in 1:L)
  _M1 = M1(T, A, B, ffnet)

  # Set up and query M2
  ρ = @variable(model)
  @constraint(model, ρ >= 0)
  _M2 = M2(ρ, ffnet)

  # Impose the LMI constraint
  M = _M1 + _M2
  @SDconstraint(model, M <= 0)

  # Objective
  @objective(model, Min, ρ)

  setup_time = time() - setup_start_time
  return model, setup_time
end

# Another equivalent way to set up the problem, for algebraic sanity
function setupViaPartialM1s(inst :: QueryInstance, opts :: LipSdpOptions)
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
  p = L - β

  # Set up the M1s
  A = sum(E(j, λdims)' * ffnet.Ms[j][1:end, 1:end-1] * E(j, mdims) for j in 1:L)
  B = sum(E(j, λdims)' * E(j+1, mdims) for j in 1:L)
  M1s = Vector{Any}()
  for k in 1:p
    Λkdim = sum(λdims[k:k+β])
    Λk = @variable(model, [1:Λkdim, 1:Λkdim], Symmetric)
    @constraint(model, Λk[1:Λkdim, 1:Λkdim] .>= 0)

    if inst.innerSparsity isa TkBanded
      Tk = Tkbanded(Λkdim, Λk, α=inst.innerSparsity.α)
    else
      error("unsupported TkSparsity: " * string(inst.innerSparsity))
    end

    Fkβ = Ec(k, β, λdims)
    Tck = Fkβ' * Tk * Fkβ
    _M1k = M1(Tck, A, B, ffnet)
    push!(M1s, _M1k)
  end

  _M1 = sum(M1s)

  # Set up and query M2
  ρ = @variable(model)
  @constraint(model, ρ >= 0)
  _M2 = M2(ρ, ffnet)

  # Impose the LMI constraint
  M = _M1 + _M2
  @SDconstraint(model, M <= 0)

  # Objective
  @objective(model, Min, ρ)

  setup_time = time() - setup_start_time
  return model, setup_time
end

#
function setupViaPartialY(inst :: QueryInstance, opts :: LipSdpOptions)
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
  p = L - β

  M1s = Vector{Any}()
  for k in 1:p
    Λkdim = sum(λdims[k:k+β])
    Λk = @variable(model, [1:Λkdim, 1:Λkdim], Symmetric)
    @constraint(model, Λk[1:Λkdim, 1:Λkdim] .>= 0)

    if inst.innerSparsity isa TkBanded
      Tk = Tkbanded(Λkdim, Λk, α=inst.innerSparsity.α)
    else
      error("unsupported TkSparsity: " * string(inst.innerSparsity))
    end

    _Yk = Y(k, β, Tk, ffnet)
    Ekβp1 = Ec(k, β+1, mdims)
    _M1k = Ekβp1' * _Yk * Ekβp1
    push!(M1s, _M1k)
  end

  _M1 = sum(M1s)

  # Set up and query M2
  ρ = @variable(model)
  @constraint(model, ρ >= 0)
  _M2 = M2(ρ, ffnet)

  # Impose the LMI constraint
  M = _M1 + _M2
  @SDconstraint(model, M <= 0)

  # Objective
  @objective(model, Min, ρ)

  setup_time = time() - setup_start_time
  return model, setup_time
end

# Depending on the set up options we call different things
function setup(inst :: QueryInstance, opts :: LipSdpOptions)
  if opts.setupMethod isa WholeTSetup
    return setupViaWholeT(inst, opts)
  elseif opts.setupMethod isa PartialM1Setup
    return setupViaPartialM1s(inst, opts)
  elseif opts.setupMethod isa PartialYSetup
    return setupViaPartialY(inst, opts)
  else
    error("unsupported setup method: " * string(opts.setupMethod))
  end
end

# Run the query and return the solution summary
function solve!(model, opts :: LipSdpOptions)
  optimize!(model)
  return solution_summary(model)
end

# The interface to call
function run(inst :: QueryInstance, opts :: LipSdpOptions)
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

#
export LipSdpOptions, WholeTSetup, PartialM1Setup, PartialYSetup
export M1, M2
export setup, solve!, run

end # End module

