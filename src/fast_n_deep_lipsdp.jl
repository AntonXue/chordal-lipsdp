module FastNDeepLipSdp

using LinearAlgebra
using Printf

include("core.jl");
include("utils.jl");

using Reexport
@reexport using .Core
@reexport using .Utils

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
  ffnet = randomNetwork(xdims)
  lopts = LipSdpOptions(τ=1, verbose=verbose)
  lsoln = solveLip(ffnet, lopts)
  copts = ChordalSdpOptions(τ=1, verbose=verbose)
  csoln = solveLip(ffnet, copts)
  if verbose; @printf("warmup time: %.3f\n", time() - warmup_start_time) end
end

export solveLip, warmup

end

