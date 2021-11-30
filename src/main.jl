
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

innerSparsity = TkBanded(α=α)

inst = QueryInstance(net=ffnet, β=β, innerSparsity=innerSparsity)

wholeTopts = LipSdpOptions(setupMethod=WholeTSetup()) 
partialM1opts = LipSdpOptions(setupMethod=PartialM1Setup()) 
partialYopts = LipSdpOptions(setupMethod=PartialYSetup())

solny = LipSdp.run(inst, partialYopts)
solnwhole = LipSdp.run(inst, wholeTopts)
solnm1 = LipSdp.run(inst, partialM1opts)

