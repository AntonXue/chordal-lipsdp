
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("lipsdp.jl"); using .LipSdp

using LinearAlgebra
using JuMP
using Random

# Fix the seed before doing anything crazy
Random.seed!(1234)

#
xdims = [2;3;4;3;4;3]
mdims = xdims[1:end-1]
λdims = mdims[2:end]

β = 1
α = 2

ffnet = randomNetwork(xdims)
K = ffnet.K
L = K - 1

p = L - β

Ldims = Vector{Int}()
Ls = Vector{Any}()
for k = 1:p
  ldim = sum(λdims[k:k+β])
  push!(Ldims, ldim)
  push!(Ls, Symmetric(randn(ldim, ldim)))
end

inst = QueryInstance(net=ffnet)

opts = QueryOptions(β=β, TkSparsity=TkαBanded(α=α))


A = sum(E(j, λdims)' * ffnet.Ms[j][1:end, 1:end-1] * E(j, mdims) for j in 1:L)
B = sum(E(j, λdims)' * E(j+1, mdims) for j in 1:L)

soln = LipSdp.run(inst, opts)

