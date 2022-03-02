start_time = time()

using LinearAlgebra
using ArgParse
using Printf
using Parameters

include("../src/FastNDeepLipSdp.jl"); using .FastNDeepLipSdp

#
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--test"
      action = :store_true
    "--nnetdir"
      arg_type = String
      required = true
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()

@printf("import loading time: %.3f\n", time() - start_time)

# Do a warmup of the stuff
println("warming up ...")
warmup(verbose=true)
println("\n")

@printf("repl start time: %.3f\n", time() - start_time)


@with_kw struct RunNNetResult
  nnet_filepath :: String
  τs :: VecInt
  lipsdp_solve_times :: VecF64
  lipsdp_total_times :: VecF64
  lipsdp_vals :: VecF64
  lipsdp_eigmaxs :: VecF64

  chordal_solve_times :: VecF64
  chordal_total_times :: VecF64
  chordal_vals :: VecF64
  chordal_eigmaxs :: VecF64
end

# The function to call for a particular nnet
function runNNet(nnet_filepath, τs, saveto_dir="~/dump"; solve_timeout=300.0)
  ffnet = loadNeuralNetwork(nnet_filepath)
  nnet_filename = basename(nnet_filepath)

  println("Running: $(nnet_filename)")

  lipsdp_solve_times, chordal_solve_times = VecF64([]), VecF64([])
  lipsdp_total_times, chordal_total_times = VecF64([]), VecF64([])
  lipsdp_vals, chordal_vals = VecF64([]), VecF64([])
  lipsdp_eigmaxs, chordal_eigmaxs = VecF64([]), VecF64([])

  for (i, τ) in enumerate(τs)
    loop_start_time = time()
    println("tick for τ[$(i)/$(length(τs))] = $(τ) of $(nnet_filename)")

    lipsdp_opts = LipSdpOptions(τ=τ, max_solve_time=solve_timeout, solver_tol=1e-6, verbose=true, use_dual=true)
    lipsdp_soln = solveLip(ffnet, lipsdp_opts)
    push!(lipsdp_solve_times, lipsdp_soln.solve_time)
    push!(lipsdp_total_times, lipsdp_soln.total_time)
    push!(lipsdp_vals, lipsdp_soln.objective_value)

    lipsdp_Z = makeZ(lipsdp_soln.values[:γ], τ, ffnet)
    eigmax_lipsdp_Z = eigmax(Symmetric(lipsdp_Z))
    push!(lipsdp_eigmaxs, eigmax_lipsdp_Z)
    @printf("\tlipsdp eigmax: %.5f\n", eigmax_lipsdp_Z)

    # Chordal stuff
    chordal_opts = ChordalSdpOptions(τ=τ, max_solve_time=solve_timeout, solver_tol=1e-6, verbose=true, use_dual=false)
    chordal_soln = solveLip(ffnet, chordal_opts)
    push!(chordal_solve_times, chordal_soln.solve_time)
    push!(chordal_total_times, chordal_soln.total_time)
    push!(chordal_vals, chordal_soln.objective_value)

    chordal_Z = makeZ(chordal_soln.values[:γ], τ, ffnet)
    eigmax_chordal_Z = eigmax(Symmetric(chordal_Z))
    push!(chordal_eigmaxs, eigmax_chordal_Z)
    @printf("\tchordal eigmax: %.5f\n", eigmax_chordal_Z)

    println("--")
  end

  # Save data
  nnet_filename = basename(nnet_filepath)
  times_saveto = "$(saveto_dir)/$(nnet_filename)_times.png"
  vals_saveto = "$(saveto_dir)/$(nnet_filename)_vals.png"

  times_title = "total times (secs) of $(nnet_filename)"
  labeled_time_lines = [("lipsdp", lipsdp_total_times); ("chordal", chordal_total_times);]
  Utils.plotLines(τs, labeled_time_lines,
                  title = times_title,
                  saveto = times_saveto)
  println("saved times info at $(times_saveto)")

  vals_title = "lipschitz upper-bounds of $(nnet_filename)"
  labeled_val_lines = [("lipsdp", lipsdp_vals); ("chordal", chordal_vals);]
  Utils.plotLines(τs, labeled_val_lines,
                  title = vals_title,
                  ylogscale = true,
                  saveto = vals_saveto)
  println("saved vals info at $(vals_saveto)")

  return RunNNetResult(
    nnet_filepath = nnet_filepath,
    τs = τs,
    lipsdp_solve_times = lipsdp_solve_times,
    lipsdp_total_times = lipsdp_total_times,
    lipsdp_vals = lipsdp_vals,
    lipsdp_eigmaxs = lipsdp_eigmaxs,
    chordal_solve_times = chordal_solve_times,
    chordal_total_times = chordal_total_times,
    chordal_vals = chordal_vals,
    chordal_eigmaxs = chordal_eigmaxs)
end

# Dim to filepath
nnet_dim_to_filepath(w, d) = "$(args["nnetdir"])/rand-I2-O2-W$(w)-D$(d).nnet"

# Run a batch
function runBatch(batch)
  for (w, d) in batch
    dim_start_time = time()
    nnet_filepath = nnet_dim_to_filepath(w, d)
    println("About to run $(nnet_filepath)")
    τs = 0:9
    runNNet(nnet_filepath, τs)
    println("done with $(nnet_filepath)")
    @printf("------------------------- time: %.3f\n", time() - dim_start_time)
  end
end

# Set up some batches for convenience
BATCH_W10 = [(10, k) for k in [5; 10; 15; 20; 25; 30; 35; 40; 45; 50]]
BATCH_W20 = [(20, k) for k in [5; 10; 15; 20; 25; 30; 35; 40;]]
BATCH_W30 = [(30, k) for k in [5; 10; 15; 20; 25; 30;]]


TEST_NNET = nnet_dim_to_filepath(10, 5)


