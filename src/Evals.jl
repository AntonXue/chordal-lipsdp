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

# Result from running a nnet
@with_kw struct RunNNetResult
  ffnet::NeuralNetwork
  weight_scales::VecF64
  solns::Vector{Any}
  lipconsts::VecF64
  others::Any
end

# A generic runnet function
function runNNet(nnet_filepath, method;
                 τs = 0:9,
                 Wk_opnorm = nothing,
                 mosek_opts = EVALS_MOSEK_OPTS)
  # The τ values are meaningful
  @assert sort(τs) == τs && minimum(τs) >= 0

  # Load the stuff and do things
  if Wk_opnorm isa Nothing
    ffnet = loadNeuralNetwork(nnet_filepath)
    weight_scales = ones(ffnet.K)
  else
    ffnet, weight_scales = loadNeuralNetwork(nnet_filepath, Wk_opnorm)
  end

  nnet_filename = basename(nnet_filepath)
  solns = Vector{Any}()
  lipconsts = VecF64()
  others = Vector{Any}()
  for (i, τ) in enumerate(τs)
    println("tick for τ[$(i)/$(length(τs))] = $(τ) of $(nnet_filename)")
    println("now: $(now) (running $(method))")

    # Run different methods depending on what is specified
    if method == :lipsdp
      opts = LipSdpOptions(τ=τ, mosek_opts=mosek_opts, verbose=true)
      soln = solveLipschitz(ffnet, opts)
      lipconst = sqrt(soln.values[:γ][end]) / prod(weight_scales)
      eigmaxZ = eigmax(Symmetric(makeZ(soln.values[:γ], τ, ffnet)))
      @printf("\teigmaxZ: %.5f \t\tlipconst: %f (%s)\n", eigmaxZ, lipconst, soln.termination_status)
      push!(others, eigmaxZ)

    elseif method == :chordalsdp
      opts = ChordalSdpOptions(τ=τ, mosek_opts=mosek_opts, verbose=true)
      soln = solveLipschitz(ffnet, opts)
      lipconst = sqrt(soln.values[:γ][end]) / prod(weight_scales)
      eigmaxZ = eigmax(Symmetric(makeZ(soln.values[:γ], τ, ffnet)))
      @printf("\teigmaxZ: %.5f \t\tlipconst: %f (%s)\n", eigmaxZ, lipconst, soln.termination_status)
      push!(others, eigmaxZ)


    else
      error("\tunrecognized method $(method)")
    end

    push!(solns, soln)
    push!(lipconsts, lipconst)
  end

  # Return the stuff
  return RunNNetResult(
    ffnet = ffnet,
    weight_scales = weight_scales,
    solns = solns,
    lipconsts = lipconsts,
    others = others)
end

# Run the lipsdp stuff
function runNNetLipSdp(nnet_filepath;
                       τs = 0:9,
                       Wk_opnorm = nothing,
                       mosek_opts = EVALS_MOSEK_OPTS,
                       saveto_dir = joinpath(homedir(), "dump"))
  res = runNNet(nnet_filepath, :lipsdp, τs=τs, Wk_opnorm=Wk_opnorm, mosek_opts=mosek_opts)
  df = DataFrame(
    τs=τs,
    lipsdp_solve_secs = [s.solve_time for s in res.solns],
    lipsdp_total_secs = [s.total_time for s in res.solns],
    lipsdp_lipconst = res.lipconsts,
    lipsdp_eigmaxZ = res.others,
    lipsdp_term_status = [s.termination_status for s in res.solns])
  nnet_filename = basename(nnet_filepath)
  saveto = joinpath(saveto_dir, "$(nnet_filename)_lipsdp.csv")
  CSV.write(saveto, df)
  println("Wrote CSV to $(saveto)")
end

# Run the chordal stuff
function runNNetChordalSdp(nnet_filepath;
                           τs = 0:9,
                           Wk_opnorm = nothing,
                           mosek_opts = EVALS_MOSEK_OPTS,
                           saveto_dir = joinpath(homedir(), "dump"))
  res = runNNet(nnet_filepath, :chordalsdp, τs=τs, Wk_opnorm=Wk_opnorm, mosek_opts=mosek_opts)
  df = DataFrame(
    τs=τs,
    chordalsdp_solve_secs = [s.solve_time for s in res.solns],
    chordalsdp_total_secs = [s.total_time for s in res.solns],
    chordalsdp_lipconst = res.lipconsts,
    chordalsdp_eigmaxZ = res.others,
    chordalsdp_term_status = [s.termination_status for s in res.solns])
  nnet_filename = basename(nnet_filepath)
  saveto = joinpath(saveto_dir, "$(nnet_filename)_chordalsdp.csv")
  CSV.write(saveto, df)
  println("Wrote CSV to $(saveto)")
end

# Run the avg lip stuff
function runNNetAvgLip(nnet_filepath; saveto_dir = joinpath(homedir(), "dump"))
  # Load the stuff and do things
  ffnet = loadNeuralNetwork(nnet_filepath)

  # Avglip simple first
  simple_opts = AvgLipOptions(use_full=false, verbose=true)
  simple_soln = solveLipschitz(ffnet, simple_opts)
  simple_lipconst = sqrt(simple_soln.objective_value)
  @printf("avglip simple lipconst: %.4f\n", simple_lipconst)

  # Avglip full next
  full_opts = AvgLipOptions(use_full=true, verbose=true)
  full_soln = solveLipschitz(ffnet, full_opts)
  full_lipconst = sqrt(full_soln.objective_value)
  full_total_time = full_soln.total_time
  @printf("avglip full lpconst: %.4f \t total_time: %.3f\n", full_lipconst, full_total_time)

  df = DataFrame(
    simple_lipconst = [simple_lipconst],
    full_lipconst = [full_lipconst],
    full_total_secs = [full_total_time])
  nnet_filename = basename(nnet_filepath)
  saveto = joinpath(saveto_dir, "$(nnet_filename)_avglip.csv")
  CSV.write(saveto, df)
  println("Wrote CSV to $(saveto)")
end

# Call the stuff
function warmup(; verbose=false)
  warmup_start_time = time()
  xdims = [2;3;3;3;3;3;3;2]
  Random.seed!(1234)
  ffnet = randomNetwork(xdims)
  lipsdp_soln = solveLipschitz(ffnet, LipSdpOptions(τ=10, mosek_opts=EVALS_MOSEK_OPTS))
  chordal_soln = solveLipschitz(ffnet, ChordalSdpOptions(τ=10, mosek_opts=EVALS_MOSEK_OPTS))
  full_soln = solveLipschitz(ffnet, AvgLipOptions())
  simple_soln = solveLipschitz(ffnet, AvgLipOptions(use_full=false))
  rand_lipconst = randomizedLipschitz(ffnet)
  if verbose
    println("lipsdp val: $(sqrt(lipsdp_soln.values[:γ][end]))")
    println("chordal val: $(sqrt(chordal_soln.values[:γ][end]))")
    println("avglip full: $(full_soln.objective_value)")
    println("avglip simple: $(simple_soln.objective_value)")
    println("randomized: $(rand_lipconst)")
  end
  if verbose; @printf("warmup time: %.3f\n", time() - warmup_start_time) end
end

#
export EVALS_MOSEK_OPTS
export RunNNetResult
export warmup, runNNet, runNNetLipSdp, runNNetChordalSdp, runNNetAvgLip

end

