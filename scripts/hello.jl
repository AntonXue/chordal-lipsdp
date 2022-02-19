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
τs = 0:7

lprimal_times = VecF64()
ldual_times = VecF64()
cprimal_times = VecF64()
cdual_times = VecF64()

lprimal_vals = VecF64()
ldual_vals = VecF64()
cprimal_vals = VecF64()
cdual_vals = VecF64()


for τ in τs
  loop_start_time = time()
  @printf("tick for τ=%d!\n", τ)

  #=
  @printf("\tbegin LipSdp (τ=%d) with primal\n", τ)
  lopts_primal = LipSdpOptions(τ=τ, max_solve_time=60.0, solver_tol=1e-4, verbose=true, use_dual=false)
  lsoln_primal = Methods.solveLip(ffnet, lopts_primal)
  push!(lprimal_times, lsoln_primal.solve_time)
  push!(lprimal_vals, lsoln_primal.objective_value)
  =#

  @printf("\n")
  @printf("\tbegin LipSdp (τ=%d) with dual\n", τ)
  lopts_dual = LipSdpOptions(τ=τ, max_solve_time=120.0, solver_tol=1e-4, verbose=true, use_dual=true)
  lsoln_dual = Methods.solveLip(ffnet, lopts_dual)
  push!(ldual_times, lsoln_dual.solve_time)
  push!(ldual_vals, lsoln_dual.objective_value)



  @printf("\n")
  @printf("\tbegin ChordalSdp (τ=%d) with primal\n", τ)
  copts_primal = ChordalSdpOptions(τ=τ, max_solve_time=120.0, solver_tol=1e-4, verbose=true, use_dual=false)
  csoln_primal = Methods.solveLip(ffnet, copts_primal)
  push!(cprimal_times, csoln_primal.solve_time)
  push!(cprimal_vals, csoln_primal.objective_value)

  #=
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
   # ("lipsdp-primal", lprimal_times);
   ("lipsdp-dual", ldual_times);
   ("chordal-primal", cprimal_times);
   # ("chordal-dual", cdual_times);
  ]
  Utils.plotLines(τs, labeled_time_lines, title=args["nnet"], saveto="~/Desktop/times.png")

labeled_val_lines =
  [
   # ("lipsdp-primal", lprimal_vals);
   ("lipsdp-dual", ldual_vals);
   ("chordal-primal", cprimal_vals);
   # ("chordal-dual", cdual_vals);
  ]
Utils.plotLines(τs, labeled_val_lines, ylogscale=true, title=args["nnet"], saveto="~/Desktop/values.png")


@printf("repl start time: %.3f\n", time() - start_time)

