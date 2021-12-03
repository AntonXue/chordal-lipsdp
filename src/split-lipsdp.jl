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
struct YsFirstSetup <: SetupMethod end
struct ζsFirstSetup <: SetupMethod end

@with_kw struct SplitLipSdpOptions
  setupMethod :: SetupMethod
  verbose :: Bool = false
end

#
function setupViaSimple(model, inst :: QueryInstance, opts :: SplitLipSdpOptions)
  setup_start_time = time()

  ffnet = inst.net
  edims = ffnet.edims
  fdims = ffnet.fdims
  L = ffnet.L
  β = inst.β
  p = inst.p

  Xs = Vector{Any}()
  for k in 1:inst.p
    Λkdim = sum(fdims[k:k+β])
    Λk = @variable(model, [1:Λkdim, 1:Λkdim], Symmetric)
    @constraint(model, Λk[1:Λkdim, 1:Λkdim] .>= 0)
    Tk = makeT(Λkdim, Λk, inst.pattern)
    Xk = makeXk(k, β, Tk, ffnet)
    push!(Xs, Xk)
  end

  ρ = @variable(model)
  @constraint(model, ρ >= 0)

  Xinit = makeXinit(β, ρ, ffnet)
  Xfinal = makeXfinal(β, ffnet)
  Ec1 = Ec(1, β+1, edims)
  Ecp = Ec(inst.p, β+1, edims)

  # Construct the big Z
  bigZ = Ec1' * Xinit * Ec1 + Ecp' * Xfinal * Ecp
  for k in 1:p
    Eck = Ec(k, β+1, edims)
    bigZ += Eck' * Xs[k] * Eck
  end

  # Make each Z partition
  Ωinv = makeΩinv(β+1, edims)
  bigZscaled = bigZ .* Ωinv
  for k in 1:p
    Eck = Ec(k, β+1, edims)
    Zk = Eck * bigZscaled * Eck'
    @SDconstraint(model, Zk <= 0)
  end

  # Set up objective and return
  @objective(model, Min, ρ)
  setup_time = time() - setup_start_time
  return model, setup_time
end

# Make a Zk
# Where ζk is a vector of [γ[k-β], ..., γ[k], ..., γ[k+b]]
function makeZk(k :: Int, β :: Int, ζk, Ωkinv :: Matrix{Float64}, ffnet :: FeedForwardNetwork, pattern :: TPattern)
  kdims, tups = makeTilingInfo(k, β+1, ffnet.edims)
  @assert length(tups) == length(ζk)
  Zktiles = Vector{Any}()
  # i is the index to access the γs, j is the relative offset from k
  for (i, (j, (slicelow, slicehigh), (insertlow, inserthigh), jdims)) in enumerate(tups)
    γi = ζk[i]
    Yi = makeYk(i, β, γi, ffnet, pattern)
    Eslice = vcat([E(l, jdims) for l in slicelow:slicehigh]...)
    Eins = vcat([E(l, kdims) for l in insertlow:inserthigh]...)
    slicedYi = Eslice * Yi * Eslice'
    push!(Zktiles, Eins' * slicedYi * Eins)
  end

  Zk = sum(Zktiles) .* Ωkinv
  return Zk
end

# Make each Zk, albeit probably inefficiently
function makeZkviaYs(k :: Int, β :: Int, Ys :: Vector{Any}, Ωkinv :: Matrix{Float64}, edims :: Vector{Int})
  @assert 1 <= k <= length(edims)

  Zk = Ys[k]
  kdims, tups = makeTilingInfo(k, β+1, edims)
  for (j, (slicelow, slicehigh), (insertlow, inserthigh), jdims) in tups
    if j == 0; continue end

    Eslice = vcat([E(i, jdims) for i in slicelow:slicehigh]...)
    slicedY = Eslice * Ys[k+j] * Eslice'
    Eins = vcat([E(i, kdims) for i in insertlow:inserthigh]...)
    Zk += Eins' * slicedY * Eins
  end

  Zk = Zk .* Ωkinv
  return Zk
end

# Another setup method
function setupViaYsFirst(model, inst :: QueryInstance, opts :: SplitLipSdpOptions)
  setup_start_time = time()
  ffnet = inst.net
  edims = ffnet.edims
  fdims = ffnet.fdims
  L = ffnet.L
  β = inst.β

  Xs = Vector{Any}()
  for k in 1:inst.p
    Λkdim = sum(fdims[k:k+β])
    Λk = @variable(model, [1:Λkdim, 1:Λkdim], Symmetric)
    @constraint(model, Λk[1:Λkdim, 1:Λkdim] .>= 0)
    Tk = makeT(Λkdim, Λk, inst.pattern)
    Xk = makeXk(k, β, Tk, ffnet)
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

  Ωinv = makeΩinv(β+1, edims)

  for k in 1:inst.p
    Eck = Ec(k, β+1, edims)
    Ωkinv = Eck * Ωinv * Eck'
    Zk = makeZkviaYs(k, β, Ys, Ωkinv, edims)
    @SDconstraint(model, Zk <= 0)
  end
  
  # Set up objective and return
  @objective(model, Min, ρ)
  setup_time = time() - setup_start_time
  return model, setup_time
end

#
function setupViaζsFirst(model, inst :: QueryInstance, opts :: SplitLipSdpOptions)
  setup_start_time = time()
  ffnet = inst.net
  edims = ffnet.edims
  fdims = ffnet.fdims
  L = ffnet.L
  β = inst.β
  p = inst.p

  γdims = Vector{Int}(zeros(p))
  for k in 1:p
    Λkdim = sum(fdims[k:k+β])
    γkdim = Λkdim^2
    if k == 1; γkdim += 1 end # To account for the ρ at γ1
    γdims[k] = γkdim
  end

  γ = @variable(model, [1:sum(γdims)])
  @constraint(model, γ[1:sum(γdims)] .>= 0)

  Ωinv = makeΩinv(β+1, edims)

  for k in 1:p
    kinds = Hcinds(k, β, γdims)
    ζk = [E(i, γdims) * γ for i in kinds]
    Eck = Ec(k, β+1, edims)
    Ωkinv = Eck * Ωinv * Eck'
    Zk = makeZk(k, β, ζk, Ωkinv, ffnet, inst.pattern)
    @SDconstraint(model, Zk <= 0)
  end

  # Set up objective and return
  @objective(model, Min, γ[1]) # ρ = γ[1]
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
  elseif opts.setupMethod isa YsFirstSetup
    return setupViaYsFirst(model, inst, opts)
  elseif opts.setupMethod isa ζsFirstSetup
    return setupViaζsFirst(model, inst, opts)
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

