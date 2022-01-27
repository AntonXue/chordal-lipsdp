
include("header.jl"); using .Header
include("common.jl"); using .Common
include("utils.jl"); using .Utils
include("lipsdp.jl"); using .LipSdp
include("admm-lipsdp.jl"); using .AdmmLipSdp
include("split-lipsdp.jl"); using .SplitLipSdp

include("tests.jl"); using .Tests

using LinearAlgebra
using JuMP
using Random

# Fix the seed before doing anything crazy
Random.seed!(1234)

# xdims = [1;1;1;1;1;1;1;1;1;1]
# xdims = [2;2;2;2;2;2]
# xdims = [2;3;4;3;4]
# xdims = [2;3;4;5;6;7;6;5;4;3;2]
# xdims = [2; 20; 20; 20; 20; 20; 20; 2]
# xdims = [2; 10; 10; 10; 10; 10; 10; 10; 10; 10; 10; 10; 2]

xdims = [2;4;6;8;6;4;2]

# xdims = [1;2;1]

edims = xdims[1:end-1]
fdims = edims[2:end]

ffnet = randomNetwork(xdims, σ=0.5)

admminst = QueryInstance(net=ffnet, β=2, pattern=BandedPattern(band=8))
admmopts = AdmmOptions(max_iters=20000, α=1, verbose=true)

params = initParams(admminst, admmopts, randomized=false)
# cache = precompute(params, admminst, admmopts)

println("")
param_hist, admmsoln = AdmmLipSdp.run(admminst, admmopts)
println("admm soln")
println(admmsoln)
println("")

# For sanity

#=
SplitSopts = SplitOptions(setupMethod=SplitLipSdp.SimpleSetup())
splitSolnS = SplitLipSdp.run(admminst, SplitSopts)
println("splitSolnS: " * string(splitSolnS))
println("")
=#

#=
SplitSopts = SplitOptions(setupMethod=SplitLipSdp.AdmmCacheSetup())
splitSolnS = SplitLipSdp.run(admminst, SplitSopts)
println("splitSolnS: " * string(splitSolnS))
println("")
=#

# Tests.testStepY(admminst, verbose=true)


