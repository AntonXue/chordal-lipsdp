module Methods

using ..Header
using ..Common
using ..LipSdp
using ..ChordalSdp
using ..Utils

using LinearAlgebra
using Printf

# Solve a problem instance depending on what kind of options we give it
function solveLip(ffnet :: FeedForwardNetwork, opts; verbose = false)
  @assert (opts isa LipSdpOptions) || (opts isa ChordalSdpOptions)
  inst = QueryInstance(ffnet=ffnet)
  if opts isa LipSdpOptions
    soln = LipSdp.run(inst, opts)
  else
    soln = ChordalSdp.run(inst, opts)
  end
  return soln
end

function warmup(; verbose = false)
  warmup_start_time = time()
  xdims = [2;3;3;3;3;3;2]
  ffnet = randomNetwork(xdims)
  lopts = LipSdpOptions(τ=2)
  lsoln = solveLip(ffnet, lopts, verbose=verbose)
  copts = ChordalSdpOptions(τ=2)
  csoln = solveLip(ffnet, copts, verbose=verbose)
  if verbose; @printf("warmup time: %.3f\n", time() - warmup_start_time) end
end


export solveLip
export warmup

end

