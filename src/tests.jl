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
    Λk = Symmetric(randn(Λkdim, Λkdim))
    Tk = makeT(Λkdim, Λk, inst.pattern)
    push!(Ts, Tk)
  end

  ρ = randn()

  # Make the simple version of M first
  T = sum(Ec(k, β, λdims)' * Ts[k] * Ec(k, β, λdims) for k in 1:inst.p)
  A = sum(E(j, λdims)' * ffnet.Ms[j][1:end, 1:end-1] * E(j, mdims) for j in 1:L)
  B = sum(E(j, λdims)' * E(j+1, mdims) for j in 1:L)

  simpleM1 = LipSdp.makeM1(T, A, B, ffnet)
  simpleM2 = LipSdp.makeM2(ρ, ffnet)
  simpleM = simpleM1 + simpleM2

  # Now make the decomposed version
  Ys = Vector{Any}()
  for k in 1:inst.p
    Yk = makeY(k, β, Ts[k], ffnet)
    push!(Ys, Yk)
  end

  Yinit = makeYinit(β, ρ, ffnet)
  Yfinal = makeYfinal(β, ffnet)

  Zs = Vector{Any}()
  for k in 1:inst.p
    Zk = Ys[k]
    if k == 1; Zk += Yinit end
    if k == inst.p; Zk += Yfinal end
    push!(Zs, Zk)
  end

  decomposedM = sum(Ec(k, β+1, mdims)' * Zs[k] * Ec(k, β+1, mdims) for k in 1:inst.p)

  Mdiff = simpleM - decomposedM
  maxdiff = maximum(abs.(Mdiff))
  println("maxdiff: " * string(maxdiff))
  @assert maxdiff <= 1e-13
end


end # End module

