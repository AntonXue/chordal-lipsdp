# Run some sanity checking test functions
module Tests

using ..Header
using ..Common
using ..LipSdp
using ..SplitLipSdp

using LinearAlgebra
using Random

# Test that two different formulations of M are equivalent
function testEquivMs(inst :: QueryInstance)
  ffnet = inst.net
  mdims = ffnet.mdims
  λdims = ffnet.λdims
  L = ffnet.L
  β = inst.β

  Ts = Vector{Any}()
  for k in 1:inst.p
    Λkdim = sum(λdims[k:k+β])
    Λk = Symmetric(abs.(randn(Λkdim, Λkdim)))
    Tk = makeT(Λkdim, Λk, inst.pattern)
    push!(Ts, Tk)
  end

  ρ = abs.(randn())

  # Make the simple version of M first
  T = sum(Ec(k, β, λdims)' * Ts[k] * Ec(k, β, λdims) for k in 1:inst.p)
  A = sum(E(j, λdims)' * ffnet.Ms[j][1:end, 1:end-1] * E(j, mdims) for j in 1:L)
  B = sum(E(j, λdims)' * E(j+1, mdims) for j in 1:L)

  simpleM1 = LipSdp.makeM1(T, A, B, ffnet)
  simpleM2 = LipSdp.makeM2(ρ, ffnet)
  simpleM = simpleM1 + simpleM2

  # Now make the decomposed version
  Xs = Vector{Any}()
  for k in 1:inst.p
    Xk = makeX(k, β, Ts[k], ffnet)
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

  decomposedM = sum(Ec(k, β+1, mdims)' * Ys[k] * Ec(k, β+1, mdims) for k in 1:inst.p)

  Mdiff = simpleM - decomposedM
  maxdiff = maximum(abs.(Mdiff))
  # println("maxdiff: " * string(maxdiff))
  @assert maxdiff <= 1e-13
  return (simpleM1, simpleM2)
end

function testΩinv(inst :: QueryInstance)
  ffnet = inst.net
  mdims = ffnet.mdims
  β = inst.β
  p = inst.p

  M1, M2 = testEquivMs(inst)
  M = M1 + M2
  Ωinv = makeΩinv(β+1, mdims)
  Mscaled = M .* Ωinv

  Zs = Vector{Any}()
  for k in 1:p
    Ekβp1 = Ec(k, β+1, mdims)
    Zk = Ekβp1 * Mscaled * Ekβp1'
    push!(Zs, Zk)
  end

  Mrecovered = sum(Ec(k, β+1, mdims)' * Zs[k] * Ec(k, β+1, mdims) for k in 1:p)
  Mdiff = M - Mrecovered
  maxdiff = maximum(abs.(Mdiff))
  println("maxdiff: " * string(maxdiff))
  @assert maxdiff <= 1e-13
end



end # End module

