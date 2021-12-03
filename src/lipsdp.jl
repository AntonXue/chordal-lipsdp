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
abstract type SetupMethod end
struct WholeTSetup <: SetupMethod end
struct SummedXSetup <: SetupMethod end
struct ScaledZSetup <: SetupMethod end

@with_kw struct LipSdpOptions
  setupMethod :: SetupMethod
  verbose :: Bool = false
end

# Compute all of T then query M1
function setupViaWholeT(model, inst :: QueryInstance, opts :: LipSdpOptions)
  setup_start_time = time()

  ffnet = inst.net
  mdims = ffnet.mdims
  λdims = ffnet.λdims
  L = ffnet.L
  β = inst.β

  # Set up the T matrix as a sum
  Ts = Vector{Any}()
  for k in 1:inst.p
    Λkdim = sum(λdims[k:k+β])
    Λk = @variable(model, [1:Λkdim, 1:Λkdim], Symmetric)
    @constraint(model, Λk[1:Λkdim, 1:Λkdim] .>= 0)
    Tk = makeT(Λkdim, Λk, inst.pattern)
    Fkβ = Ec(k, β, ffnet.λdims)
    push!(Ts, Fkβ' * Tk * Fkβ)
  end

  # Query M1
  T = sum(Ts)
  A = sum(E(j, λdims)' * ffnet.Ms[j][1:end, 1:end-1] * E(j, mdims) for j in 1:L)
  B = sum(E(j, λdims)' * E(j+1, mdims) for j in 1:L)
  M1 = makeM1(T, A, B, ffnet)

  # Set up and query M2
  ρ = @variable(model)
  @constraint(model, ρ >= 0)
  M2 = makeM2(ρ, ffnet)

  # Impose the LMI constraint and objective
  M = M1 + M2
  @SDconstraint(model, M <= 0)
  @objective(model, Min, ρ)

  setup_time = time() - setup_start_time
  return model, setup_time
end

#
function setupViaSummedX(model, inst :: QueryInstance, opts :: LipSdpOptions)
  setup_start_time = time()

  ffnet = inst.net
  mdims = ffnet.mdims
  λdims = ffnet.λdims
  L = ffnet.L
  β = inst.β

  Xs = Vector{Any}()
  for k in 1:inst.p
    Λkdim = sum(λdims[k:k+β])
    Λk = @variable(model, [1:Λkdim, 1:Λkdim], Symmetric)
    @constraint(model, Λk[1:Λkdim, 1:Λkdim] .>= 0)
    Tk = makeT(Λkdim, Λk, inst.pattern)
    Xk = makeX(k, β, Tk, ffnet)
    push!(Xs, Xk)
  end

  # Set up and query M2
  ρ = @variable(model)
  @constraint(model, ρ >= 0)
  Xinit = makeXinit(β, ρ, ffnet)
  Xfinal = makeXfinal(β, ffnet)
  E1βp1 = Ec(1, β+1, mdims)
  Epβp1 = Ec(inst.p, β+1, mdims)

  # Construct the big Z
  bigZ = E1βp1' * Xinit * E1βp1 + Epβp1' * Xfinal * Epβp1
  for k in 1:inst.p
    Ekβp1 = Ec(k, β+1, mdims)
    bigZ += Ekβp1' * Xs[k] * Ekβp1
  end

  # The LMI and objective
  @SDconstraint(model, bigZ <= 0)
  @objective(model, Min, ρ)

  setup_time = time() - setup_start_time
  return model, setup_time
end

function setupViaScaledZ(model, inst :: QueryInstance, opts :: LipSdpOptions)
  setup_start_time = time()

  ffnet = inst.net
  mdims = ffnet.mdims
  λdims = ffnet.λdims
  L = ffnet.L
  β = inst.β

  Xs = Vector{Any}()
  for k in 1:inst.p
    Λkdim = sum(λdims[k:k+β])
    Λk = @variable(model, [1:Λkdim, 1:Λkdim], Symmetric)
    @constraint(model, Λk[1:Λkdim, 1:Λkdim] .>= 0)
    Tk = makeT(Λkdim, Λk, inst.pattern)
    Xk = makeX(k, β, Tk, ffnet)
    push!(Xs, Xk)
  end

  # Set up and query M2
  ρ = @variable(model)
  @constraint(model, ρ >= 0)
  Xinit = makeXinit(β, ρ, ffnet)
  Xfinal = makeXfinal(β, ffnet)
  E1βp1 = Ec(1, β+1, mdims)
  Epβp1 = Ec(inst.p, β+1, mdims)

  # Construct the big Z
  bigZ = E1βp1' * Xinit * E1βp1 + Epβp1' * Xfinal * Epβp1
  for k in 1:inst.p
    Ekβp1 = Ec(k, β+1, mdims)
    bigZ += Ekβp1' * Xs[k] * Ekβp1
  end

  Zs = Vector{Any}()
  Ωinv = makeΩinv(β+1, mdims)
  bigZscaled = bigZ .* Ωinv
  for k in 1:inst.p
    Ekβp1 = Ec(k, β+1, mdims)
    Zk = Ekβp1 * bigZscaled * Ekβp1'
    push!(Zs, Zk)
  end

  bigZagain = sum(Ec(k, β+1, mdims)' * Zs[k] * Ec(k, β+1, mdims) for k in 1:inst.p)

  @SDconstraint(model, bigZagain <= 0)
  @objective(model, Min, ρ)

  setup_time = time() - setup_start_time
  return model, setup_time
end

# Depending on the set up options we call different things
function setup(inst :: QueryInstance, opts :: LipSdpOptions)
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

  if opts.setupMethod isa WholeTSetup
    return setupViaWholeT(model, inst, opts)
  elseif opts.setupMethod isa SummedXSetup
    return setupViaSummedX(model, inst, opts)
  elseif opts.setupMethod isa ScaledZSetup
    return setupViaScaledZ(model, inst, opts)
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
export LipSdpOptions, WholeTSetup, SummedXSetup
export setup, solve!, run

end # End module

