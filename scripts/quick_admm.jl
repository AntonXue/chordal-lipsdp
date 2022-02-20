start_time = time()

using LinearAlgebra
using ArgParse
using Printf

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

inst = QueryInstance(ffnet=ffnet)
admm_opts = AdmmSdpOptions(τ=2)

init_params, _ = initAdmmParams(inst, admm_opts)

cache, _ = precomputeAdmmCache(inst, init_params, admm_opts)

