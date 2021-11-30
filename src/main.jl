
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("lipsdp.jl"); using .LipSdp
include("split-lipsdp.jl"); using .SplitLipSdp

include("tests.jl"); using .Tests

using LinearAlgebra
using JuMP
using Random

# Fix the seed before doing anything crazy
Random.seed!(1234)

#
# xdims = [2;3;4;3;4;3]
xdims = [2;3;4;5;6;7;6;5;4;3;2]
mdims = xdims[1:end-1]
λdims = mdims[2:end]

β = 3
α = 6

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

pattern = BandedPattern(band=α)

inst = QueryInstance(net=ffnet, β=β, pattern=pattern)

#
wholeTopts = LipSdpOptions(setupMethod=WholeTSetup()) 
partialM1opts = LipSdpOptions(setupMethod=PartialM1Setup()) 
partialYopts = LipSdpOptions(setupMethod=PartialYSetup())

solny = LipSdp.run(inst, partialYopts)
solnw = LipSdp.run(inst, wholeTopts)
solnm = LipSdp.run(inst, partialM1opts)

#
splitopts = SplitLipSdpOptions()
solnsplit = SplitLipSdp.run(inst, splitopts)

