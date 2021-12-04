
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("lipsdp.jl"); using .LipSdp
include("split-lipsdp.jl"); using .SplitLipSdp
include("admm-lipsdp.jl"); using .AdmmLipSdp

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
# xdims = [2; 10; 10; 10; 10; 10; 10; 10; 10; 10; 10; 10; 2]
# xdims = [2; 20; 20; 20; 20; 20; 20; 2]
edims = xdims[1:end-1]
fdims = edims[2:end]

ffnet = randomNetwork(xdims, σ=0.5)

reginst = QueryInstance(net=ffnet, β=1, pattern=OnePerNeuronPattern())


# Simple testing

#=
SimpleTopts = LipSdpOptions(setupMethod=LipSdp.WholeTSetup()) 
solnT = LipSdp.run(reginst, SimpleTopts)
println("solnT: " * string(solnT))
println("")

SimpleXopts = LipSdpOptions(setupMethod=LipSdp.SummedXSetup())
solnX = LipSdp.run(reginst, SimpleXopts)
println("solnX: " * string(solnX))
println("")

SimpleZopts = LipSdpOptions(setupMethod=LipSdp.ScaledZSetup())
solnZ = LipSdp.run(reginst, SimpleZopts)
println("solnZ: " * string(solnZ))
println("")
=#

# Split testing

splitinst = QueryInstance(net=ffnet, β=1, pattern=BandedPattern(band=2))

#=
SplitSopts = SplitOptions(setupMethod=SplitLipSdp.SimpleSetup())
splitSolnS = SplitLipSdp.run(splitinst, SplitSopts)
println("splitSolnS: " * string(splitSolnS))
println("")

SplitYopts = SplitOptions(setupMethod=SplitLipSdp.YsFirstSetup())
splitSolnY = SplitLipSdp.run(splitinst, SplitYopts)
println("splitSolnY: " * string(splitSolnY))
println("")

Splitζopts = SplitOptions(setupMethod=SplitLipSdp.ζsFirstSetup())
splitSolnζ = SplitLipSdp.run(splitinst, Splitζopts)
println("splitSolnζ: " * string(splitSolnζ))
println("")
=#

# ADMM testing


admminst = QueryInstance(net=ffnet, β=1, pattern=BandedPattern(band=2))
admmopts = AdmmOptions(verbose=true)
params = initParams(admminst, admmopts)

cache = precompute(params, admminst, admmopts)

println("\n\n\n")

admmsoln = AdmmLipSdp.run(admminst, admmopts)



