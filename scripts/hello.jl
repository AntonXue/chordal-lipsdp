start_time = time()
include("../src/core/header.jl"); using .Header
include("../src/core/common.jl"); using .Common
include("../src/core/lipsdp.jl"); using .LipSdp
include("../src/core/chordalsdp.jl"); using .ChordalSdp
include("../src/nnet-parser.jl"); using .NNetParser
include("../src/utils.jl"); using .Utils
include("../src/methods.jl"); using .Methods

using LinearAlgebra
using ArgParse
using Printf

#
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--test"
      action = :store_true
    "--nnet"
      arg_type = String
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

if !(args["nnet"] isa Nothing) && isfile(args["nnet"])
  ffnet = loadFeedForwardNetwork(args["nnet"])
end


