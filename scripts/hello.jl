hello_start_time = time()
using LinearAlgebra
using SparseArrays
using ArgParse
using Printf
using Parameters
using MosekTools
using Dates

include("../src/Evals.jl"); using .Evals
@printf("import loading time: %.3f\n", time() - hello_start_time)

# Argument parsing
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--nnet"
      help = "A particular nnet that is loaded"
    "--nnetdir"
      help = "Directory of the nnet files"
      # required = true
      default = "nnets"
    "--dumpdir"
      help = "Directory of where to dump things"
      default = joinpath(homedir(), "dump")
    "--skipwarmup"
      help = "Do not do the warmups"
      action = :store_true
  end
  return parse_args(ARGS, argparse_settings)
end
args = parseArgs()

# Set up some constants
NNET_DIR = args["nnetdir"]; @assert isdir(NNET_DIR)
RAND_NNET_DIR = joinpath(NNET_DIR, "rand"); @assert isdir(RAND_NNET_DIR)
ACAS_NNET_DIR = joinpath(NNET_DIR, "acas"); @assert isdir(ACAS_NNET_DIR)
DUMP_DIR = args["dumpdir"]; @assert isdir(DUMP_DIR)

# Some batches of random networks
rand_nnet_filepath(w, d) = "$(RAND_NNET_DIR)/rand-I2-O2-W$(w)-D$(d).nnet"
RAND_W10_BATCH = [(10, d, rand_nnet_filepath(10, d)) for d in [5; 10; 15; 20; 25; 30; 35; 40; 45; 50]]
RAND_W20_BATCH = [(20, d, rand_nnet_filepath(20, d)) for d in [5; 10; 15; 20; 25; 30; 35; 40; 45; 50]]
RAND_W30_BATCH = [(30, d, rand_nnet_filepath(30, d)) for d in [5; 10; 15; 20; 25; 30; 35; 40; 45; 50]]
RAND_W40_BATCH = [(40, d, rand_nnet_filepath(40, d)) for d in [5; 10; 15; 20; 25; 30; 35; 40; 45; 50]]
RAND_W50_BATCH = [(50, d, rand_nnet_filepath(50, d)) for d in [5; 10; 15; 20; 25; 30; 35; 40; 45; 50]]
SMALL_RAND_BATCH = [(10, d, rand_nnet_filepath(10, d)) for d in [5; 10; 15]]

# The ACAS networks
ACAS_FILES = readdir(ACAS_NNET_DIR, join=true)
ACAS1_BATCH = filter(f -> (match(r".*run2a_1.*nnet", f) isa RegexMatch), ACAS_FILES)
ACAS2_BATCH = filter(f -> (match(r".*run2a_2.*nnet", f) isa RegexMatch), ACAS_FILES)
ACAS3_BATCH = filter(f -> (match(r".*run2a_3.*nnet", f) isa RegexMatch), ACAS_FILES)
ACAS4_BATCH = filter(f -> (match(r".*run2a_4.*nnet", f) isa RegexMatch), ACAS_FILES)
ACAS5_BATCH = filter(f -> (match(r".*run2a_5.*nnet", f) isa RegexMatch), ACAS_FILES)
SMALL_ACAS_BATCH = ACAS_FILES[1:2]

# Run a batch of random networks
function runRandBatch(rand_batch;
                      τs = 0:9,
                      lipsdp_mosek_opts = EVALS_MOSEK_OPTS,
                      chordalsdp_mosek_opts = EVALS_MOSEK_OPTS,
                      target_opnorm = nothing)
  batch_size = length(rand_batch)
  results = Vector{Any}()
  for (i, (w, d, nnet_filepath)) in enumerate(rand_batch)
    iter_start_time = time()
    println("About to run rand [$(i)/$(batch_size)]: $(w)W $(d)D")
    rand_saveto_dir = joinpath(DUMP_DIR, "rand")
    res = runNNet(
      nnet_filepath,
      τs = τs, # There is an opportunity to smartly scale
      lipsdp_mosek_opts = lipsdp_mosek_opts,
      chordalsdp_mosek_opts = chordalsdp_mosek_opts,
      target_opnorm = target_opnorm,
      saveto_dir = rand_saveto_dir)
    push!(results, res)
    @printf("----------- iter done in time: %.3f\n", time() - iter_start_time)
  end
  return results
end

# Run an ACAS batch
function runAcasBatch(acas_batch;
                      τs = 0:6,
                      lipsdp_mosek_opts = EVALS_MOSEK_OPTS,
                      chordalsdp_mosek_opts = EVALS_MOSEK_OPTS)
  batch_size = length(acas_batch)
  results = Vector{RunNNetResult}()
  for (i, acas_filepath) in enumerate(acas_batch)
    iter_start_time = time()
    println("About to run ACAS [$(i)/$(batch_size)]: $(acas_filepath)")
    acas_saveto_dir = joinpath(DUMP_DIR, "acas")
    res = runNNet(
      acas_filepath,
      τs = τs,
      lipsdp_mosek_opts = lipsdp_mosek_opts,
      chordalsdp_mosek_opts = chordalsdp_mosek_opts,
      saveto_dir = acas_saveto_dir)
    push!(results, res)
    @printf("----------- iter done in time: %.3f\n", time() - iter_start_time)
  end
  return results
end

# Run a batch
TEST_NNET_FILEPATH = rand_nnet_filepath(10, 5)
test_ffnet = loadNeuralNetwork(TEST_NNET_FILEPATH)

# Do a warmup of the stuff
if !args["skipwarmup"]
  println("warming up ...")
  warmup(verbose=false)
end

@printf("repl start time: %.3f\n", time() - hello_start_time)

unscaled_ffnet = loadNeuralNetwork(args["nnet"])
scaled_ffnet, weight_scales = loadNeuralNetwork(args["nnet"], 2.0)


