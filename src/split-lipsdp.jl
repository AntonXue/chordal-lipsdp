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
function setupViaSimple(model, inst :: QueryInstance, opts :: SplitLipSdpOptions)
  setup_start_time = time()

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
# function makeZ(k :: Int, β :: Int, γ, γdims :: Vector{Int}, mdims)
function makeZ(k :: Int, β :: Int, Ys :: Vector{Any}, Ωkinv :: Matrix{Float64}, mdims :: Vector{Int})
  @assert 1 <= k <= length(mdims)

  Zk = Ys[k]
  kdims, tups = makePartitionTuples(k, β+1, mdims)
  for (j, (slicelow, slicehigh), (inslow, inshigh), jdims) in tups
    if j == 0; continue end

    Eslice = vcat([E(i, jdims) for i in slicelow:slicehigh]...)
    
    # println("Ys[" * string(k+j) * "]: " * string(size(Ys[k+j])))
    # println("Eslice size: " * string(size(Eslice)))

    slicedY = Eslice * Ys[k+j] * Eslice'

    # println("slicedY size: " * string(size(slicedY)))
    Eins = vcat([E(i, kdims) for i in inslow:inshigh]...)
    # println("Eins size: " * string(size(Eins)))
    Zk += Eins' * slicedY * Eins
  end

  Zk = Zk .* Ωkinv
  return Zk
end

# Another setup method
function setupViaΓ(model, inst :: QueryInstance, opts :: SplitLipSdpOptions)
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

  ρ = @variable(model)
  @constraint(model, ρ >= 0)
  Xinit = makeXinit(β, ρ, ffnet)
  Xfinal = makeXfinal(β, ffnet)

  Ys = Vector{Any}()
  for k in 1:inst.p
    Yk = Xs[k]
    if k == 1; Yk += Xinit end
    if k == inst.p; Yk += Xfinal end
    push!(Ys, Yk)
  end

  Ωinv = makeΩinv(β+1, mdims)

  for k in 1:inst.p
    Eck = Ec(k, β+1, mdims)
    Ωkinv = Eck * Ωinv * Eck'
    Zk = makeZ(k, β, Ys, Ωkinv, mdims)
    @SDconstraint(model, Zk <= 0)
  end
  
  # Set up objective and return
  @objective(model, Min, ρ)
  setup_time = time() - setup_start_time
  return model, setup_time
end

# Depending on the options, call the appropriate setup function
function setup(inst :: QueryInstance, opts :: SplitLipSdpOptions)
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

  if opts.setupMethod isa SimpleSetup
    return setupViaSimple(model, inst, opts)
  elseif opts.setupMethod isa ΓSetup
    return setupViaΓ(model, inst, opts)
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

