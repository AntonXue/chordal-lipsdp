# Implmenetation of a split version of LipSdp
module SplitLipSdp

using ..Header
using ..Common
using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

# How the construction is done
abstract type SetupMethod end
struct SimpleSetup <: SetupMethod end
struct ΓSetup <: SetupMethod end

@with_kw struct SplitLipSdpOptions
  setupMethod :: SetupMethod
  verbose :: Bool = false
end

#
function setupViaSimple(inst :: QueryInstance, opts :: SplitLipSdpOptions)
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

  Xs = Vector{Any}()
  for k in 1:inst.p
    Λkdim = sum(λdims[k:k+β])
    Λk = @variable(model, [1:Λkdim, 1:Λkdim], Symmetric)
    @constraint(model, Λk[1:Λkdim, 1:Λkdim] .>= 0)
    Tk = makeT(Λkdim, Λk, inst.pattern)
    Xk = makeX(k, β, Tk, ffnet)
    push!(Xs, Xk)
  end

  ρ = @variable(model)
  @constraint(model, ρ >= 0)

  Xinit = makeXinit(β, ρ, ffnet)
  Xfinal = makeXfinal(β, ffnet)
  E1βp1 = Ec(1, β+1, mdims)
  Epβp1 = Ec(inst.p, β+1, mdims)

  # Construct the big Z
  bigZ = E1βp1' * Xinit * E1βp1 + Epβp1' * Xfinal * Epβp1
  for k in 1:p
    Ekβp1 = Ec(k, β+1, mdims)
    bigZ += Ekβp1' * Xs[k] * Ekβp1
  end

  # Make each Z partition
  Ωinv = makeΩinv(β+1, mdims)
  bigZscaled = bigZ .* Ωinv
  for k in 1:p
    Ekβp1 = Ec(k, β+1, mdims)
    Zk = Ekβp1 * bigZscaled * Ekβp1'
    @SDconstraint(model, Zk <= 0)
  end

  # Set up objective and return
  @objective(model, Min, ρ)
  setup_time = time() - setup_start_time
  return model, setup_time
end

# Make each Zk, albeit probably inefficiently
function makeZ(k :: Int, ζk, γdims :: Vector{Int}, inst :: QueryInstance)
  β = inst.β
  kdims, tups = makePartitionTuples(k, β, γdims)
  lentup = length(tups)

  Xs = Vector{Any}()
end

# Another setup method
function setupViaΓ(inst :: QueryInstance, opts :: SplitLipSdpOptions)
  #=
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


  γdims = Vector{Int}(zeros(L))
  @variable(model, γ[1:sum(γdims)] >= 0)
  for k in 1:L
    ζk = Hc(k, γdims) * γ
    Zk = makeZ(k, ζk, γdims, ...)
    @SDconstraint(model, Zk <= 0)
  end

  return model
  =#
end

# Depending on the options, call the appropriate setup function
function setup(inst :: QueryInstance, opts :: SplitLipSdpOptions)
  if opts.setupMethod isa SimpleSetup
    return setupViaSimple(inst, opts)
  elseif opts.setupMethod isa ΓSetup
    return setupViaΓ(inst, opts)
  else
    error("unsupported setup method: " * string(opts.setupMethod))
  end
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
export makeZ

end # End module

