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

ffnet = loadFeedForwardNetwork(args["nnet"])

# Do a warmup of the stuff
Methods.warmup(verbose=true)

# Different betas to try
# βs = 0:4
βs = 0:39

ltimes = VecF64()
lvals = VecF64()

ctimes = VecF64()
cvals = VecF64()

for β in βs
  loop_start_time = time()
  @printf("tick for β=%d!\n", β)
  lopts = LipSdpOptions(β=β)
  lsoln = Methods.solveLip(ffnet, lopts)
  push!(ltimes, lsoln.solve_time)
  push!(lvals, lsoln.objective_value)
  @printf("\tlipsdp \ttime: %.3f, \tvalue: %.3f\n", lsoln.solve_time, lsoln.objective_value)

  copts = ChordalSdpOptions(β=β)
  csoln = Methods.solveLip(ffnet, copts)
  push!(ctimes, csoln.solve_time)
  push!(cvals, csoln.objective_value)
  @printf("\tchordl \ttime: %.3f, \tvalue: %.3f\n", csoln.solve_time, csoln.objective_value)
end

# Plot stuff

labeled_time_lines = [("lipsdp", ltimes); ("chordal", ctimes)]
Utils.plotLines(βs, labeled_time_lines, saveto="~/Desktop/times.png")

labeled_val_lines = [("lipsdp", lvals); ("chordal", cvals)]
Utils.plotLines(βs, labeled_val_lines, ylogscale=true, saveto="~/Desktop/values.png")


@printf("repl start time: %.3f\n", time() - start_time)

