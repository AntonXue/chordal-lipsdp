# A bunch of helpful stuff for running evaluations
module Evals

using LinearAlgebra
using Printf
using Random

include("FastNDeepLipSdp.jl"); using .FastNDeepLipSdp


function warmup(; verbose=false)
  warmup_start_time = time()
  xdims = [2;3;3;3;3;3;3;2]
  Random.seed!(1234)
  ffnet = randomNetwork(xdims)
  lipsdp_opts = LipSdpOptions(τ=1, verbose=verbose, use_dual=true)
  lipsdp_soln = solveLip(ffnet, lipsdp_opts)
  chordal_opts = ChordalSdpOptions(τ=1, verbose=verbose)
  chordal_soln = solveLip(ffnet, chordal_opts)
  if verbose; @printf("warmup time: %.3f\n", time() - warmup_start_time) end
end

export warmup

end
