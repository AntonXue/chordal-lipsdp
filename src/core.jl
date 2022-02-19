module Core

include("core/header.jl");
include("core/common.jl");
include("core/lipsdp.jl");
include("core/chordalsdp.jl");

# Type definitions in core/header.jl
export VecInt, VecF64, MatF64
export Activation, ReluActivation, TanhActivation, NeuralNetwork
export QueryInstance, SolutionOutput

# Common funtionalities in core/common.jl
export e, E, Ec
export makeA, makeB
export γlength, makeT, makeM1, makeM2
export makeCliqueInfos

# Method-specific types and polymorphic functions
export LipSdpOptions, ChordalSdpOptions
export setup!, solve!, runQuery

end
