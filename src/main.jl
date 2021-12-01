
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
# xdims = [2;3;4;3;4]
# xdims = [2;3;4;5;6;7;6;5;4;3;2]
# xdims = [2; 20; 20; 20; 20; 20; 20; 2]
xdims = [2; 10; 10; 10; 10; 10; 10; 10; 2]
mdims = xdims[1:end-1]
λdims = mdims[2:end]

numneurons = sum(xdims)

ffnet = randomNetwork(xdims, σ=0.8)

reginst = QueryInstance(net=ffnet, β=1, pattern=OnePerNeuronPattern())
splitinst = QueryInstance(net=ffnet, β=3, pattern=BandedPattern(band=8))

#
wholeTopts = LipSdpOptions(setupMethod=WholeTSetup()) 
partialM1opts = LipSdpOptions(setupMethod=PartialM1Setup()) 
partialYopts = LipSdpOptions(setupMethod=PartialYSetup())

solny = LipSdp.run(reginst, partialYopts)
solnw = LipSdp.run(reginst, wholeTopts)
solnm = LipSdp.run(reginst, partialM1opts)

#
splitopts = SplitLipSdpOptions()
solnsplit = SplitLipSdp.run(splitinst, splitopts)

