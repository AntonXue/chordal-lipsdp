module Stuff

include("Stuff/header.jl");
include("Stuff/common.jl");
include("Stuff/lipsdp.jl");
include("Stuff/chordalsdp.jl");
include("Stuff/admmsdp.jl");

# Type definitions in core/header.jl
export VecInt, VecF64, MatF64, SpVecInt, SpVecF64, SpMatF64
export Activation, ReluActivation, TanhActivation, NeuralNetwork
export QueryInstance, SolutionOutput

# Common funtionalities in core/common.jl
export e, E, Ec
export makeA, makeB
export γlength, makeT, makeM1, makeM2, makeZ
export makeCliqueInfos

# Method-specific types and polymorphic functions
export LipSdpOptions, ChordalSdpOptions
export setup!, solve!, runQuery

# ADMM-specific stuff
export AdmmSdpOptions, AdmmStatus, AdmmSummary, AdmmParams, AdmmCache
export initAdmmParams, initAdmmCache

end
