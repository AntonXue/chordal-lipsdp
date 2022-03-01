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

# Do a warmup of the stuff
@printf("warming up ...\n")
warmup(verbose=true)
@printf("\n\n")

# Different betas to try
# τs = 0:4
τs = 0:10

lip_solve_times = VecF64([])
chord_solve_times = VecF64([])

lip_total_times = VecF64([])
chord_total_times = VecF64([])

lip_vals = VecF64([])
chord_vals = VecF64([])


for τ in τs
  loop_start_time = time()
  @printf("tick for τ=%d!\n", τ)

  @printf("\n")
  @printf("\tbegin LipSdp (τ=%d) with dual\n", τ)
  lip_opts = LipSdpOptions(τ=τ, max_solve_time=120.0, solver_tol=1e-6, verbose=true, use_dual=true)
  lip_soln = solveLip(ffnet, lip_opts)
  push!(lip_solve_times, lip_soln.solve_time)
  push!(lip_total_times, lip_soln.total_time)
  push!(lip_vals, lip_soln.objective_value)

  @printf("\n")
  @printf("\tbegin ChordalSdp (τ=%d) with primal\n", τ)
  chord_opts = ChordalSdpOptions(τ=τ, max_solve_time=120.0, solver_tol=1e-6, verbose=true, use_dual=false)
  chord_soln = solveLip(ffnet, chord_opts)
  push!(chord_solve_times, chord_soln.solve_time)
  push!(chord_total_times, chord_soln.total_time)
  push!(chord_vals, chord_soln.objective_value)

  @printf("\n--\n\n")
end

# Plot stuff

nnet_filename = basename(args["nnet"])
times_saveto = "~/dump/$(nnet_filename)_times.png"
vals_saveto= "~/dump/$(nnet_filename)_vals.png"

labeled_time_lines =
  [
   ("lipsdp", lip_total_times);
   ("chordal", chord_total_times);
  ]
  Utils.plotLines(τs, labeled_time_lines, title=args["nnet"], saveto=times_saveto)

labeled_val_lines =
  [
   ("lipsdp", lip_vals);
   ("chordal", chord_vals);
  ]
Utils.plotLines(τs, labeled_val_lines, ylogscale=true, title=args["nnet"], saveto=vals_saveto)


@printf("repl start time: %.3f\n", time() - start_time)

