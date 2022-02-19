start_time = time()

using LinearAlgebra
using ArgParse
using Printf

include("../src/fast_n_deep_lipsdp.jl"); using .FastNDeepLipSdp

#
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--test"
      action = :store_true
    "--nnet"
      arg_type = String
      required = true
    #=
    "--deepsdp"
      action = :store_true
    "--splitsdp"
      action = :store_true
    "--benchdir"
      help = "the NNet file location"
      arg_type = String
      required = true
    "--tband"
      arg_type = Int
    =#
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()

@printf("import loading time: %.3f\n", time() - start_time)

ffnet = loadNeuralNetwork(args["nnet"])


