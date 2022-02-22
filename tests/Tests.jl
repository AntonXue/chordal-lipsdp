# Run some sanity checking test functions
module Tests

using LinearAlgebra
using Random
using JuMP
using MosekTools

include("../src/FastNDeepLipSdp.jl")

import Reexport
Reexport.@reexport using .FastNDeepLipSdp

include("test_admm_cache.jl");


end # End module

using .Tests

