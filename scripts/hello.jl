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

# Set up some constants; set up some directories
NNET_DIR = args["nnetdir"]; @assert isdir(NNET_DIR)
RAND_NNET_DIR = joinpath(NNET_DIR, "rand"); @assert isdir(RAND_NNET_DIR)
ACAS_NNET_DIR = joinpath(NNET_DIR, "acas"); @assert isdir(ACAS_NNET_DIR)
DUMP_DIR = args["dumpdir"]; @assert isdir(DUMP_DIR)
RAND_SAVETO_DIR = joinpath(DUMP_DIR, "rand"); isdir(RAND_SAVETO_DIR) || mkdir(RAND_SAVETO_DIR)
ACAS_SAVETO_DIR = joinpath(DUMP_DIR, "acas"); isdir(ACAS_SAVETO_DIR) || mkdir(ACAS_SAVETO_DIR)

# Some batches of random networks
rand_nnet_filepath(w, d) = "$(RAND_NNET_DIR)/rand-I2-O2-W$(w)-D$(d).nnet"
RAND_W10_BATCH = [(rand_nnet_filepath(10, d), 0:15, 2.0) for d in [5; 10; 15; 20; 25; 30; 35; 40; 45; 50]]
RAND_W20_BATCH = [(rand_nnet_filepath(20, d), 0:15, 2.0) for d in [5; 10; 15; 20; 25; 30; 35; 40; 45; 50]]
RAND_W30_BATCH = [(rand_nnet_filepath(30, d), 0:15, 1.8) for d in [5; 10; 15; 20; 25; 30; 35; 40; 45; 50]]
RAND_W40_BATCH = [(rand_nnet_filepath(40, d), 0:9,  1.7) for d in [5; 10; 15; 20; 25; 30; 35; 40; 45; 50]]
RAND_W50_BATCH = [(rand_nnet_filepath(50, d), 0:6,  1.6) for d in [5; 10; 15; 20; 25; 30; 35; 40; 45; 50]]
SMALL_RAND_BATCH = [(rand_nnet_filepath(10, d), 0:3, 2.0) for d in [5; 10; 15]]

# The ACAS networks
ACAS_FILES = readdir(ACAS_NNET_DIR, join=true)
ALL_ACAS_BATCH = [(f, 0:9, 2.0) for f in ACAS_FILES]
ACAS1_BATCH = filter(b -> (match(r".*run2a_1.*nnet", b[1]) isa RegexMatch), ALL_ACAS_BATCH)
ACAS2_BATCH = filter(b -> (match(r".*run2a_2.*nnet", b[1]) isa RegexMatch), ALL_ACAS_BATCH)
ACAS3_BATCH = filter(b -> (match(r".*run2a_3.*nnet", b[1]) isa RegexMatch), ALL_ACAS_BATCH)
ACAS4_BATCH = filter(b -> (match(r".*run2a_4.*nnet", b[1]) isa RegexMatch), ALL_ACAS_BATCH)
ACAS5_BATCH = filter(b -> (match(r".*run2a_5.*nnet", b[1]) isa RegexMatch), ALL_ACAS_BATCH)
SMALL_ACAS_BATCH = [(f, 0:2, 2.0) for f in ACAS_FILES]

# Run a batch
function runBatch(batch, method, saveto_dir; mosek_opts = EVALS_MOSEK_OPTS)
  batch_size = length(batch)
  results = Vector{Any}()
  for (i, (nnet_filepath, τs, Wk_opnorm)) in enumerate(batch)
    iter_start_time = time()
    println("About to run [$(i)/$(batch_size)]: $(nnet_filepath)")
    if method == :lipsdp
      runNNetLipSdp(nnet_filepath, τs=τs, Wk_opnorm=Wk_opnorm, mosek_opts=mosek_opts, saveto_dir=saveto_dir)
    elseif method == :chordalsdp
      runNNetChordalSdp(nnet_filepath, τs=τs, Wk_opnorm=Wk_opnorm, mosek_opts=mosek_opts, saveto_dir=saveto_dir)
    elseif method == :avglip
      runNNetAvgLip(nnet_filepath, saveto_dir=saveto_dir)
    else
      error("unrecognized method: $(method)")
    end
    @printf("----------- iter done in time: %.3f\n", time() - iter_start_time)
  end
  return results
end

# Shortcut for rand
function runRandBatch(batch, method; mosek_opts = EVALS_MOSEK_OPTS)
  return runBatch(batch, method, RAND_SAVETO_DIR, mosek_opts=mosek_opts)
end

# Shortcut for Acas
function runAcasBatch(batch, method; mosek_opts = EVALS_MOSEK_OPTS)
  return runBatch(batch, method, ACAS_SAVETO_DIR, mosek_opts=mosek_opts)
end

# Do a warmup of the stuff
if !args["skipwarmup"]
  println("warming up ...")
  warmup(verbose=true)
end

@printf("repl start time: %.3f\n", time() - hello_start_time)

ffnet = loadNeuralNetwork(args["nnet"])
unscaled_ffnet = loadNeuralNetwork(args["nnet"])
scaled_ffnet, weight_scales = loadNeuralNetwork(args["nnet"], 2.0)


