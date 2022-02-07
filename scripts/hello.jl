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
@printf("\n\n")

# Different betas to try
# τs = 0:4
τs = 0:4

ltimes = VecF64()
lvals = VecF64()

ctimes = VecF64()
cvals = VecF64()

for τ in τs
  loop_start_time = time()
  @printf("tick for τ=%d!\n", τ)

  @printf("begin LipSdp (τ=%d)\n", τ)
  lopts = LipSdpOptions(τ=τ, max_solve_time=120.0, solver_tol=1e-4, verbose=true)
  lsoln = Methods.solveLip(ffnet, lopts)
  push!(ltimes, lsoln.solve_time)
  push!(lvals, lsoln.objective_value)
  @printf("\tlipsdp \ttime: %.3f, \tvalue: %.3f (%s)\n", lsoln.solve_time, lsoln.objective_value, lsoln.termination_status)
  # push!(ltimes, 10)
  # push!(lvals, 10)
  
  @printf("\n\t--\n\n")

  @printf("begin ChordalSdp (τ=%d)\n", τ)
  copts = ChordalSdpOptions(τ=τ, max_solve_time=120.0, solver_tol=1e-4, verbose=true)
  csoln = Methods.solveLip(ffnet, copts)
  push!(ctimes, csoln.solve_time)
  push!(cvals, csoln.objective_value)
  @printf("\tchordl \ttime: %.3f, \tvalue: %.3f (%s)\n", csoln.solve_time, csoln.objective_value, csoln.termination_status)

  @printf("\n\n--\n\n")
end

# Plot stuff

labeled_time_lines = [("lipsdp", ltimes); ("chordal", ctimes)]
Utils.plotLines(τs, labeled_time_lines, saveto="~/Desktop/times.png")

labeled_val_lines = [("lipsdp", lvals); ("chordal", cvals)]
Utils.plotLines(τs, labeled_val_lines, ylogscale=true, saveto="~/Desktop/values.png")


@printf("repl start time: %.3f\n", time() - start_time)

