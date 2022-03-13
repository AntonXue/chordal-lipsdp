module FastNDeepLipSdp

using LinearAlgebra
using Printf
using Random

include("Stuff/Stuff.jl");
include("Utils/Utils.jl");

import Reexport
Reexport.@reexport using .Stuff
Reexport.@reexport using .Utils

# Solve a problem instance depending on what kind of options we give it
function solveLip(ffnet::NeuralNetwork, opts::SdpOptions; verbose = false)
  inst = QueryInstance(ffnet=ffnet)
  soln = runQuery(inst, opts) # Multiple dispatch based on opts type
  return soln
end

export solveLip

end

