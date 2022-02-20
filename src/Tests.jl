# Run some sanity checking test functions
module Tests

using LinearAlgebra
using Random
using JuMP
using MosekTools

include("FastNDeepLipSdp.jl")

import Reexport
Reexport.@reexport using .FastNDeepLipSdp

include("Tests/admm_cache_tests.jl");


end # End module

using .Tests

