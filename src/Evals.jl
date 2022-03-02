# A bunch of helpful stuff for running evaluations
module Evals

using LinearAlgebra
using Printf
using Random
using Parameters

include("FastNDeepLipSdp.jl")

import Reexport
Reexport.@reexport using .FastNDeepLipSdp

# Default options for Mosek
EVALS_DEFAULT_MOSEK_OPTS = Dict(
  "QUIET" => true,
  "MSK_DPAR_OPTIMIZER_MAX_TIME" => 600.0,
  "INTPNT_CO_TOL_REL_GAP" => 1e-6,
  "INTPNT_CO_TOL_PFEAS" => 1e-6,
  "INTPNT_CO_TOL_DFEAS" => 1e-6
)

# Call the stuff
function warmup(; verbose=false)
  warmup_start_time = time()
  xdims = [2;3;3;3;3;3;3;2]
  Random.seed!(1234)
  ffnet = randomNetwork(xdims)
  lipsdp_opts = LipSdpOptions(τ=1, mosek_opts=EVALS_DEFAULT_MOSEK_OPTS, verbose=verbose, use_dual=true)
  lipsdp_soln = solveLip(ffnet, lipsdp_opts)
  chordal_opts = ChordalSdpOptions(τ=1, mosek_opts=EVALS_DEFAULT_MOSEK_OPTS, verbose=verbose)
  chordal_soln = solveLip(ffnet, chordal_opts)
  if verbose; @printf("warmup time: %.3f\n", time() - warmup_start_time) end
end

# Results from a runNNet function call
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
function runNNet(nnet_filepath;
                 τs = 0:9,
                 lipsdp_mosek_opts = EVALS_DEFAULT_MOSEK_OPTS,
                 chordalsdp_mosek_opts = EVALS_DEFAULT_MOSEK_OPTS,
                 saveto_dir = "~/dump")
  @assert sort(τs) == τs && minimum(τs) >= 0
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

    # LipSdp stuff
    lipsdp_opts = LipSdpOptions(τ=τ, mosek_opts=lipsdp_mosek_opts, verbose=true, use_dual=true)
    lipsdp_soln = solveLip(ffnet, lipsdp_opts)
    lipsdp_Z = makeZ(lipsdp_soln.values[:γ], τ, ffnet)
    eigmax_lipsdp_Z = eigmax(Symmetric(lipsdp_Z))
    @printf("\tlipsdp eigmax: %.6f\n", eigmax_lipsdp_Z)

    push!(lipsdp_solve_times, lipsdp_soln.solve_time)
    push!(lipsdp_total_times, lipsdp_soln.total_time)
    push!(lipsdp_vals, lipsdp_soln.objective_value)
    push!(lipsdp_eigmaxs, eigmax_lipsdp_Z)

    # Chordal stuff
    chordal_opts = ChordalSdpOptions(τ=τ, mosek_opts=chordalsdp_mosek_opts, verbose=true)
    chordal_soln = solveLip(ffnet, chordal_opts)
    chordal_Z = makeZ(chordal_soln.values[:γ], τ, ffnet)
    eigmax_chordal_Z = eigmax(Symmetric(chordal_Z))
    @printf("\tchordal eigmax: %.5f\n", eigmax_chordal_Z)

    push!(chordal_solve_times, chordal_soln.solve_time)
    push!(chordal_total_times, chordal_soln.total_time)
    push!(chordal_vals, chordal_soln.objective_value)
    push!(chordal_eigmaxs, eigmax_chordal_Z)

    println("--")
  end

  # Save the time stuff
  times_saveto = "$(saveto_dir)/$(nnet_filename)_times.png"
  times_title = "total times (secs) of $(nnet_filename)"
  labeled_time_lines = [("lipsdp", lipsdp_total_times); ("chordal", chordal_total_times);]
  Utils.plotLines(τs, labeled_time_lines,
                  title = times_title,
                  saveto = times_saveto)
  println("saved times info at $(times_saveto)")

  # Save the lipschitz constant plots
  vals_saveto = "$(saveto_dir)/$(nnet_filename)_vals.png"
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

#
export RunNNetResult
export warmup, runNNet

end

