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
function solveLip(ffnet :: NeuralNetwork, opts; verbose = false)
  @assert (opts isa LipSdpOptions) || (opts isa ChordalSdpOptions)
  inst = QueryInstance(ffnet=ffnet)
  soln = runQuery(inst, opts) # Multiple dispatch based on opts type
  return soln
end

function warmup(; verbose=false)
  warmup_start_time = time()
  xdims = [2;3;3;3;3;3;2]
  Random.seed!(1234)
  ffnet = randomNetwork(xdims)
  lipsdp_opts = LipSdpOptions(τ=1, verbose=verbose, use_dual=true)
  lipsdp_soln = solveLip(ffnet, lipsdp_opts)
  chordal_opts = ChordalSdpOptions(τ=1, verbose=verbose)
  chordal_soln = solveLip(ffnet, chordal_opts)
  if verbose; @printf("warmup time: %.3f\n", time() - warmup_start_time) end
end

export solveLip, warmup

end

