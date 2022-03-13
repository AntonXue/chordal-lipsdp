# A bunch of helpful stuff for running evaluations
module Evals

using LinearAlgebra
using Printf
using Random
using Parameters
using DataFrames
using CSV
using Dates

include("FastNDeepLipSdp.jl")
import Reexport
Reexport.@reexport using .FastNDeepLipSdp

# Default options for Mosek
EVALS_MOSEK_OPTS =
  Dict("QUIET" => true,
       # "MSK_IPAR_INTPNT_SCALING" => 3,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 2,
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

# Call the stuff
function warmup(; verbose=false)
  warmup_start_time = time()
  xdims = [2;3;3;3;3;3;3;2]
  Random.seed!(1234)
  ffnet = randomNetwork(xdims)
  lipsdp_opts = LipSdpOptions(τ=1, mosek_opts=EVALS_MOSEK_OPTS)
  lipsdp_soln = solveLip(ffnet, lipsdp_opts)
  chordal_opts = ChordalSdpOptions(τ=1, mosek_opts=EVALS_MOSEK_OPTS)
  chordal_soln = solveLip(ffnet, chordal_opts)
  if verbose; @printf("warmup time: %.3f\n", time() - warmup_start_time) end
end

# Results from a runNNet function call
@with_kw struct RunNNetResult
  nnet_filepath::String
  τs::VecInt

  lipsdp_solve_times::VecF64
  lipsdp_total_times::VecF64
  lipsdp_vals::VecF64
  lipsdp_eigmaxs::VecF64
  lipsdp_term_statuses::Vector{String}

  chordal_solve_times::VecF64
  chordal_total_times::VecF64
  chordal_vals::VecF64
  chordal_eigmaxs::VecF64
  chordal_term_statuses::Vector{String}
end

function saveRunNNetResult(res::RunNNetResult, saveto)
  # Construct the DataFrame
  df = DataFrame(
    taus = res.τs,
    lipsdp_solve_secs = res.lipsdp_solve_times,
    lipsdp_total_secs = res.lipsdp_total_times,
    lipsdp_lipschitz_vals = res.lipsdp_vals,
    lipsdp_Z_eigmaxs = res.lipsdp_eigmaxs,
    lipsdp_term_statuses = res.lipsdp_term_statuses,
    chordal_solve_secs = res.chordal_solve_times,
    chordal_total_secs = res.chordal_total_times,
    chordal_lipschitz_vals = res.chordal_vals,
    chordal_Z_eigmaxs = res.chordal_eigmaxs,
    chordal_term_statuses = res.chordal_term_statuses)
  CSV.write(saveto, df)
  println("Wrote CSV to $(saveto)")
end

# The function to call for a particular nnet
function runNNet(nnet_filepath;
                 τs = 0:9,
                 lipsdp_mosek_opts = EVALS_MOSEK_OPTS,
                 chordalsdp_mosek_opts = EVALS_MOSEK_OPTS,
                 saveto_dir = joinpath(homedir(), "dump"),
                 target_opnorm = 2.0,
                 do_plots = false,
                 profile_stuff = false) # TODO: implement profiling

  # The τ values are meaningful
  @assert sort(τs) == τs && minimum(τs) >= 0

  # Make directory if it doesn't exist yet
  isdir(saveto_dir) || mkdir(saveto_dir)

  # Load the stuff and do things
  if target_opnorm isa Nothing
    ffnet = loadNeuralNetwork(nnet_filepath)
    weight_scales = ones(ffnet.K)
  else
    ffnet, weight_scales = loadNeuralNetwork(nnet_filepath, target_opnorm)
  end

  # raw_ffnet = loadNeuralNetwork(nnet_filepath)
  # ffnet = scaleNeuralNetwork(raw_ffnet, weight_scale)
  nnet_filename = basename(nnet_filepath)

  lipsdp_solve_times, chordal_solve_times = VecF64(), VecF64()
  lipsdp_total_times, chordal_total_times = VecF64(), VecF64()
  lipsdp_vals, chordal_vals = VecF64(), VecF64()
  lipsdp_eigmaxs, chordal_eigmaxs = VecF64(), VecF64()
  lipsdp_term_statuses, chordal_term_statuses = Vector{String}(), Vector{String}()

  for (i, τ) in enumerate(τs)
    println("tick for τ[$(i)/$(length(τs))] = $(τ) of $(nnet_filename)")

    # Chordal stuff
    println("now: $(now()) (running chordal!)")

    chordal_opts = ChordalSdpOptions(τ=τ, mosek_opts=chordalsdp_mosek_opts, verbose=true)
    chordal_soln = solveLip(ffnet, chordal_opts)
    chordal_lipconst = sqrt(chordal_soln.objective_value) / prod(weight_scales)
    chordal_Z = makeZ(chordal_soln.values[:γ], τ, ffnet)
    eigmax_chordal_Z = eigmax(Symmetric(chordal_Z))
    println("\tchordal eigmax: $(eigmax_chordal_Z) \tlipconst: $(chordal_lipconst)")

    push!(chordal_solve_times, chordal_soln.solve_time)
    push!(chordal_total_times, chordal_soln.total_time)
    push!(chordal_vals, chordal_lipconst)
    push!(chordal_eigmaxs, eigmax_chordal_Z)
    push!(chordal_term_statuses, chordal_soln.termination_status)


    #=
    # LipSdp stuff
    println("now: $(now()) (running lipsdp!)")
    lipsdp_opts = LipSdpOptions(τ=τ, mosek_opts=lipsdp_mosek_opts, verbose=true)
    lipsdp_soln = solveLip(ffnet, lipsdp_opts)
    lipsdp_lipconst = sqrt(lipsdp_soln.objective_value) / prod(weight_scales)
    lipsdp_Z = makeZ(lipsdp_soln.values[:γ], τ, ffnet)
    eigmax_lipsdp_Z = eigmax(Symmetric(lipsdp_Z))
    println("\tlipsdp eigmax: $(eigmax_lipsdp_Z) \tlipconst: $(lipsdp_lipconst)")

    push!(lipsdp_solve_times, lipsdp_soln.solve_time)
    push!(lipsdp_total_times, lipsdp_soln.total_time)
    push!(lipsdp_vals, lipsdp_lipconst)
    push!(lipsdp_eigmaxs, eigmax_lipsdp_Z)
    push!(lipsdp_term_statuses, lipsdp_soln.termination_status)
    =#

    println("--")
  end

  # Save the time stuff
  if do_plots
    times_saveto = "$(saveto_dir)/$(nnet_filename)_times.png"
    times_title = "total times (secs) of $(nnet_filename)"
    labeled_time_lines = [("lipsdp", lipsdp_total_times); ("chordal", chordal_total_times);]
    Utils.plotLines(τs, labeled_time_lines, title=times_title, saveto=times_saveto)
    println("saved times info at $(times_saveto)")

    # Save the lipschitz constant plots
    vals_saveto = "$(saveto_dir)/$(nnet_filename)_vals.png"
    vals_title = "lipschitz upper-bounds of $(nnet_filename)"
    labeled_val_lines = [("lipsdp", lipsdp_vals); ("chordal", chordal_vals);]
    Utils.plotLines(τs, labeled_val_lines, title=vals_title, ylogscale=true, saveto=vals_saveto)
    println("saved vals info at $(vals_saveto)")
  end

  res = RunNNetResult(
    nnet_filepath = nnet_filepath,
    τs = τs,
    lipsdp_solve_times = lipsdp_solve_times,
    lipsdp_total_times = lipsdp_total_times,
    lipsdp_vals = lipsdp_vals,
    lipsdp_eigmaxs = lipsdp_eigmaxs,
    lipsdp_term_statuses = lipsdp_term_statuses,
    chordal_solve_times = chordal_solve_times,
    chordal_total_times = chordal_total_times,
    chordal_vals = chordal_vals,
    chordal_eigmaxs = chordal_eigmaxs,
    chordal_term_statuses = chordal_term_statuses)

  # Save the CSV
  # csv_saveto = joinpath(saveto_dir, "$(nnet_filename)_runs.csv")
  # saveRunNNetResult(res, csv_saveto)
  return res
end

#
export EVALS_MOSEK_OPTS
export RunNNetResult
export warmup, runNNet

end

