start_time = time()

using LinearAlgebra
using ArgParse
using Printf
using Parameters

include("../src/Evals.jl"); using .Evals

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


TEST_NNET_FILEPATH = nnet_dim_to_filepath(10, 5)
test_ffnet = loadNeuralNetwork(TEST_NNET_FILEPATH)


MOSEK_OPTS = Dict(
    "QUIET" => false,
    "MSK_DPAR_OPTIMIZER_MAX_TIME" => 300.0,
    "INTPNT_CO_TOL_REL_GAP" => 1e-5
   )



