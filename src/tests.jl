# Run some sanity checking test functions
module Tests

using ..Header
using ..Common
using ..LipSdp
using ..AdmmLipSdp
using ..SplitLipSdp

using LinearAlgebra
using Random
using JuMP
using Mosek
using MosekTools


# Test that two different formulations of M are equivalent
function testEquivMs(inst :: QueryInstance; verbose :: Bool = true)
  ffnet = inst.net
  edims = ffnet.edims
  fdims = ffnet.fdims
  L = ffnet.L
  β = inst.β

  Ts = Vector{Any}()
  for k in 1:inst.p
    Λkdim = sum(fdims[k:k+β])
    Λk = Symmetric(abs.(randn(Λkdim, Λkdim)))
    Tk = makeT(Λkdim, Λk, inst.pattern)
    push!(Ts, Tk)
  end

  # Make the simple version of M first
  T = sum(Ec(k, β, fdims)' * Ts[k] * Ec(k, β, fdims) for k in 1:inst.p)
  A = sum(E(j, fdims)' * ffnet.Ms[j][1:end, 1:end-1] * E(j, edims) for j in 1:L)
  B = sum(E(j, fdims)' * E(j+1, edims) for j in 1:L)

  simpleM1 = makeM1(T, A, B, ffnet)
  ρ = abs.(randn())
  simpleM2 = makeM2(ρ, ffnet)
  simpleM = simpleM1 + simpleM2

  # Now make the decomposed version
  Xs = Vector{Any}()
  for k in 1:inst.p
    Xk = makeXk(k, β, Ts[k], ffnet)
    push!(Xs, Xk)
  end

  Xinit = makeXinit(β, ρ, ffnet)
  Xfinal = makeXfinal(β, ffnet)

  Ys = Vector{Any}()
  for k in 1:inst.p
    Yk = Xs[k]
    if k == 1; Yk += Xinit end
    if k == inst.p; Yk += Xfinal end
    push!(Ys, Yk)
  end

  decomposedM = sum(Ec(k, β+1, edims)' * Ys[k] * Ec(k, β+1, edims) for k in 1:inst.p)

  Mdiff = simpleM - decomposedM
  maxdiff = maximum(abs.(Mdiff))

  if verbose; println("maxdiff: " * string(maxdiff)) end
  @assert maxdiff <= 1e-13
  return (simpleM1, simpleM2)
end

function testΩinv(inst :: QueryInstance; verbose :: Bool = true)
  ffnet = inst.net
  edims = ffnet.edims
  β = inst.β
  p = inst.p

  M1, M2 = testEquivMs(inst, verbose=false)
  M = M1 + M2
  Ωinv = makeΩinv(β+1, edims)
  Mscaled = M .* Ωinv

  Zs = Vector{Any}()
  for k in 1:p
    Eck = Ec(k, β+1, edims)
    Zk = Eck * Mscaled * Eck'
    push!(Zs, Zk)
  end

  Mrecovered = sum(Ec(k, β+1, edims)' * Zs[k] * Ec(k, β+1, edims) for k in 1:p)
  Mdiff = M - Mrecovered
  maxdiff = maximum(abs.(Mdiff))

  if verbose; println("maxdiff: " * string(maxdiff)) end
  @assert maxdiff <= 1e-13
end

# Test the make Zk functionality
function testZPartitions(inst; verbose :: Bool = true)
  ffnet = inst.net
  edims = ffnet.edims
  fdims = ffnet.fdims
  L = ffnet.L
  β = inst.β

  Ts = Vector{Any}()
  for k in 1:inst.p
    Λkdim = sum(fdims[k:k+β])
    Λk = Symmetric(abs.(randn(Λkdim, Λkdim)))
    Tk = makeT(Λkdim, Λk, inst.pattern)
    push!(Ts, Tk)
  end

  Xs = Vector{Any}()
  for k in 1:inst.p
    Xk = makeXk(k, β, Ts[k], ffnet)
    push!(Xs, Xk)
  end

  ρ = abs.(randn())
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

  bigZ = zeros(sum(edims), sum(edims))

  for k in 1:inst.p
    Eck = Ec(k, β+1, edims)
    Ωkinv = Eck * Ωinv * Eck'
    Zk = SplitLipSdp.makeZkviaYs(k, β, Ys, Ωkinv, edims)
    bigZ += Eck' * Zk * Eck
  end

  # Make Z using the M method
  T = sum(Ec(k, β, fdims)' * Ts[k] * Ec(k, β, fdims) for k in 1:inst.p)
  A = sum(E(j, fdims)' * ffnet.Ms[j][1:end, 1:end-1] * E(j, edims) for j in 1:L)
  B = sum(E(j, fdims)' * E(j+1, edims) for j in 1:L)
  M1 = makeM1(T, A, B, ffnet)
  M2 = makeM2(ρ, ffnet)
  M = M1 + M2

  #
  Mdiff = M - bigZ
  maxdiff = maximum(abs.(Mdiff))

  if verbose; println("maxdiff: " * string(maxdiff)) end
  @assert maxdiff <= 1e-13
end


# Test the tiling
function testTiling(inst :: QueryInstance; verbose :: Bool = true)
  ffnet = inst.net
  edims = ffnet.edims
  fdims = ffnet.fdims
  L = ffnet.L
  β = inst.β
  p = inst.p

  γdims = Vector{Int}()
  Λs = Vector{Any}()
  Ts = Vector{Any}()
  γs = Vector{Vector{Float64}}()
  for k in 1:p
    Λkdim = sum(fdims[k:k+β])
    γkdim = Λkdim^2
    if k == 1; γkdim += 1 end
    push!(γdims, γkdim)

    Λk = abs.(Symmetric(randn(Λkdim, Λkdim)))
    push!(Λs, Λk)

    γk = vec(Λk)
    push!(γs, γk)

    Tk = makeT(Λkdim, Λk, inst.pattern)
    push!(Ts, Tk)
  end

  # Form the complicated way
  ρ = abs(randn())
  γ = vcat(γs...)
  γ = [ρ; γ]
  Ωinv = makeΩinv(β+1, edims)

  bigZ = zeros(sum(edims), sum(edims))
  for k in 1:p
    overlappedYinds = Hcinds(k, β+1, γdims)

    ζk = [E(i, γdims) * γ for i in overlappedYinds]
    Eck = Ec(k, β+1, edims)
    Ωkinv = Eck * Ωinv * Eck'
    Zk = SplitLipSdp.makeZk(k, β, ζk, Ωkinv, ffnet, inst.pattern)
    bigZ += Eck' * Zk * Eck
  end

  # Make Z using the M method
  T = sum(Ec(k, β, fdims)' * Ts[k] * Ec(k, β, fdims) for k in 1:inst.p)
  A = sum(E(j, fdims)' * ffnet.Ms[j][1:end, 1:end-1] * E(j, edims) for j in 1:L)
  B = sum(E(j, fdims)' * E(j+1, edims) for j in 1:L)
  M1 = makeM1(T, A, B, ffnet)
  M2 = makeM2(ρ, ffnet)
  M = M1 + M2

  #
  Mdiff = M - bigZ
  maxdiff = maximum(abs.(Mdiff))

  if verbose; println("maxdiff: " * string(maxdiff)) end
  @assert maxdiff <= 1e-13
end


# Test the different ADMM steps
function testStepY(inst :: QueryInstance; verbose :: Bool = true)
  admmopts = AdmmOptions(verbose=true)
  params = initParams(inst, admmopts, randomized=true)
  cache = precompute(params, inst, admmopts)

  # Manual implementation of the minimization problem via a solver call
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-9))

  β = inst.β
  γdims = params.γdims

  ρ = @variable(model)
  ζs = Vector{Any}()
  for k in 1:inst.p
    ζk = @variable(model, [1:length(params.ζs[k])])
    push!(ζs, ζk)
  end

  ρterm = (ρ - params.γ[1] + (params.η / params.α))^2
  γnorms = [sum((ζs[k] - Hc(k, β+1, γdims) * params.γ + (params.μs[k] / params.α)).^2) for k in 1:inst.p]
  Vnorms = [sum((params.vs[k] - AdmmLipSdp.makezk(k, ζs[k], cache) + (params.τs[k] / params.α)).^2) for k in 1:inst.p]

  L = ρ + (params.α / 2) * (ρterm + sum(γnorms) + sum(Vnorms))
  @objective(model, Min, L)
  optimize!(model)

  solver_ρ = value.(ρ)
  solver_ζs = [value.(ζs[k]) for k in 1:inst.p]


  # The version implemented
  step_ρ, step_ζs = AdmmLipSdp.stepY(params, cache)

  ρmaxdiff = abs(solver_ρ - step_ρ)
  ζsmaxdiffs = [maximum(abs.(solver_ζs[k] - step_ζs[k])) for k in 1:inst.p]


  println("ρmaxdiff: " * string(ρmaxdiff))
  println("ζsmaxdiffs: " * string(ζsmaxdiffs'))

  return (solver_ρ, solver_ζs, step_ρ, step_ζs)
end

end # End module

