start_time = time()
include("core/header.jl"); using .Header
include("core/common.jl"); using .Common
include("core/lipsdp.jl"); using .LipSdp
include("core/chordalsdp.jl"); using .ChordalSdp
include("utils.jl"); using .Utils

using LinearAlgebra
using JuMP
using Random
using Printf

@printf("imports done: %.3f\n", time() - start_time)


# Fix the seed before doing anything crazy
Random.seed!(1234)

# xdims = [1;1;1;1;1;1;1;1;1;1]
# xdims = [2;2;2;2;2;2]
# xdims = [2;3;4;3;4]
# xdims = [2;3;4;5;6;7;6;5;4;3;2]
# xdims = [2; 20; 20; 20; 20; 20; 20; 2]
# xdims = [2; 10; 10; 10; 10; 10; 10; 10; 10; 10; 10; 10; 2]

xdims = [2;4;6;8;6;4;2]
ffnet = randomNetwork(xdims, σ=0.5)
inst = QueryInstance(ffnet=ffnet)

#
copts = ChordalSdpOptions(β=1)
csoln = ChordalSdp.run(inst, copts)

#
lopts = LipSdpOptions(β=1)
lsoln = LipSdp.run(inst, lopts)



