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
@printf("warming up ...\n")
Methods.warmup(verbose=true)
@printf("\n\n")

# Different betas to try
# τs = 0:4
τs = 3:10

lnodual_times = VecF64()
ldual_times = VecF64()
cnodual_times = VecF64()
cdual_times = VecF64()

lnodual_vals = VecF64()
ldual_vals = VecF64()
cnodual_vals = VecF64()
cdual_vals = VecF64()


for τ in τs
  loop_start_time = time()
  @printf("tick for τ=%d!\n", τ)

  #=
  @printf("\tbegin LipSdp (τ=%d) with NO dual\n", τ)
  lopts_nodual = LipSdpOptions(τ=τ, max_solve_time=60.0, solver_tol=1e-4, verbose=true, use_dual=false)
  lsoln_nodual = Methods.solveLip(ffnet, lopts_nodual)
  push!(lnodual_times, lsoln_nodual.solve_time)
  push!(lnodual_vals, lsoln_nodual.objective_value)
  =#

  #=
  @printf("\n")
  @printf("\tbegin LipSdp (τ=%d) with dual\n", τ)
  lopts_dual = LipSdpOptions(τ=τ, max_solve_time=300.0, solver_tol=1e-4, verbose=true, use_dual=true)
  lsoln_dual = Methods.solveLip(ffnet, lopts_dual)
  push!(ldual_times, lsoln_dual.solve_time)
  push!(ldual_vals, lsoln_dual.objective_value)
  =#


  @printf("\n")
  @printf("\tbegin LipSdp (τ=%d) with CDCS\n", τ)
  lopts_cdcs = LipSdpOptions(τ=τ, max_solve_time=300.0, solver_tol=1e-4, verbose=true, use_cdcs=true)
  lsoln_cdcs = Methods.solveLip(ffnet, lopts_dual)
  # push!(ldual_times, lsoln_dual.solve_time)
   #push!(ldual_vals, lsoln_dual.objective_value)

  #=
  @printf("\n")
  @printf("\tbegin ChordalSdp (τ=%d) with NO dual\n", τ)
  copts_nodual = ChordalSdpOptions(τ=τ, max_solve_time=60.0, solver_tol=1e-4, verbose=true, use_dual=false)
  csoln_nodual = Methods.solveLip(ffnet, copts_nodual)
  push!(cnodual_times, csoln_nodual.solve_time)
  push!(cnodual_vals, csoln_nodual.objective_value)

  @printf("\n")
  @printf("\tbegin ChordalSdp (τ=%d) with dual\n", τ)
  copts_dual = ChordalSdpOptions(τ=τ, max_solve_time=60.0, solver_tol=1e-4, verbose=true, use_dual=true)
  csoln_dual = Methods.solveLip(ffnet, copts_dual)
  push!(cdual_times, csoln_dual.solve_time)
  push!(cdual_vals, csoln_dual.objective_value)
  =#

  @printf("\n--\n\n")
end

# Plot stuff

labeled_time_lines =
  [
   ("lipsdp-nodual", lnodual_times);
   ("lipsdp-dual", ldual_times);
   ("chordal-nodual", cnodual_times);
   ("chordal-dual", cdual_times);
  ]
Utils.plotLines(τs, labeled_time_lines, saveto="~/Desktop/times.png")

labeled_val_lines =
  [
   ("lipsdp-nodual", lnodual_vals);
   ("lipsdp-dual", ldual_vals);
   ("chordal-nodual", cnodual_vals);
   ("chordal-dual", cdual_vals);
  ]
Utils.plotLines(τs, labeled_val_lines, ylogscale=true, saveto="~/Desktop/values.png")


@printf("repl start time: %.3f\n", time() - start_time)

