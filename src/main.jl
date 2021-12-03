
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
# xdims = [1;1;1;1;1;1;1;1;1;1]
# xdims = [2;3;4;3;4]
xdims = [2;3;4;5;6;7;6;5;4;3;2]
# xdims = [2; 20; 20; 20; 20; 20; 20; 2]
# xdims = [2; 10; 10; 10; 10; 10; 10; 10; 2]
mdims = xdims[1:end-1]
λdims = mdims[2:end]

numneurons = sum(xdims)

ffnet = randomNetwork(xdims, σ=0.8)

reginst = QueryInstance(net=ffnet, β=2, pattern=OnePerNeuronPattern())
splitinst = QueryInstance(net=ffnet, β=3, pattern=BandedPattern(band=8))

#
SimpleTopts = LipSdpOptions(setupMethod=LipSdp.WholeTSetup()) 
SimpleXopts = LipSdpOptions(setupMethod=LipSdp.SummedXSetup())
SimpleZopts = LipSdpOptions(setupMethod=LipSdp.ScaledZSetup())

#=
solnT = LipSdp.run(reginst, SimpleTopts)
println("solnT: " * string(solnT))
println("")

solnX = LipSdp.run(reginst, SimpleXopts)
println("solnX: " * string(solnX))
println("")

solnZ = LipSdp.run(reginst, SimpleZopts)
println("solnZ: " * string(solnZ))
println("")
=#

#

SplitSopts = SplitLipSdpOptions(setupMethod=SplitLipSdp.SimpleSetup())
SplitZopts = SplitLipSdpOptions(setupMethod=SplitLipSdp.ΓSetup())

splitSolnS = SplitLipSdp.run(splitinst, SplitSopts)
println("splitSolnS: " * string(splitSolnS))
println("")

splitSolnZ = SplitLipSdp.run(splitinst, SplitZopts)
println("splitSolnZ: " * string(splitSolnZ))
println("")



#=
splitopts = SplitLipSdpOptions()
solnsplit = SplitLipSdp.run(splitinst, splitopts)
=#

