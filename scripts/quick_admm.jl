start_time = time()

using LinearAlgebra
using ArgParse
using Printf
using SparseArrays

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
Zdim = sum(ffnet.edims)
Tdim = sum(ffnet.fdims)
K = ffnet.K
E1 = E(1, ffnet.edims)
EK = E(ffnet.K, ffnet.edims)

Ws = [M[1:end, 1:end-1] for M in ffnet.Ms]
A, B = makeA(ffnet), makeB(ffnet)

inst = QueryInstance(ffnet=ffnet)
τ = 2

# Load this just in case
lip_opts = LipSdpOptions(τ=τ, verbose=true)
lip_soln = runQuery(inst, lip_opts)
lipZ = makeZ(lip_soln.values[:γ], lip_opts.τ, ffnet)
lipγ = lip_soln.values[:γ]
@printf("lipγρ: %.3f\n", lipγ[end])

@printf("\n\n")

# init_params, init_time = initAdmmParams(inst, admm_opts)
admm_opts = AdmmSdpOptions(τ=τ, verbose=true, max_steps=500, ρ=1)
admm_soln = runQuery(inst, admm_opts)
admm_params = admm_soln.values
admm_cache, _ = initAdmmCache(inst, admm_params, admm_opts)

num_cliques = admm_params.num_cliques
cinfos = admm_params.cinfos

admmZ = makeZ(admm_soln.values.γ, admm_opts.τ, ffnet)
psdZ = -1 * Symmetric(reshape(Stuff.projectNsd(-vec(admmZ)), size(admmZ)))
Ecs = [Ec(k, Ckdim, Zdim) for (k, _, Ckdim) in cinfos]
manualZs = [Symmetric(reshape(admm_params.zs[k], (Ckdim, Ckdim))) for (k, _, Ckdim) in cinfos]
manualZ = sum(Ecs[k]' * manualZs[k] * Ecs[k] for k in 1:num_cliques)

# manualVs = [Symmetric(reshape(admm_params.vs[k], (Ckdim, Ckdim))) for (k, _, Ckdim) in cinfos]
# manualV = sum(Ecs[k]' * manualVs[k] * Ecs[k] for k in 1:num_cliques)

err_primal_hist, err_dual_hist = admm_soln.summary.err_hist

