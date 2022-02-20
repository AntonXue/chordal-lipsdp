start_time = time()

using LinearAlgebra
using ArgParse
using Printf
using SparseArrays

include("../src/FastNDeepLipSdp.jl"); using .FastNDeepLipSdp

#
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--test"
      action = :store_true
    "--nnet"
      arg_type = String
      required = true
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()

@printf("import loading time: %.3f\n", time() - start_time)

ffnet = loadNeuralNetwork(args["nnet"])
Zdim = sum(ffnet.edims)
Tdim = sum(ffnet.fdims)

inst = QueryInstance(ffnet=ffnet)
admm_opts = AdmmSdpOptions(τ=2, verbose=true, max_steps=200)

# Load this just in case
lip_opts = LipSdpOptions(τ=admm_opts.τ, verbose=true)
lip_soln = runQuery(inst, lip_opts)
lipZ = makeZ(lip_soln.values[:γ], lip_opts.τ, ffnet)

# init_params, init_time = initAdmmParams(inst, admm_opts)
# cache, cache_time = initAdmmCache(inst, init_params, admm_opts)
admm_soln = runQuery(inst, admm_opts)

Z = makeZ(admm_soln.values.γ, admm_opts.τ, ffnet)

psdZ = -1 * Symmetric(reshape(Stuff.projectNsd(-vec(Z)), size(Z)))

