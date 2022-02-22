using Parameters
using LinearAlgebra
using SparseArrays
using JuMP
using MosekTools
using Printf

# Options for ADMM
@with_kw struct AdmmSdpOptions
  τ :: Int = 0
  ρ :: Float64 = 1.0
  max_steps :: Int = 200
  
  max_solver_X_time :: Float64 = 60.0
  solver_X_tol :: Float64 = 1e-4
  cholesky_reg_ε :: Float64 = 1e-2
  verbose :: Bool = false
end

# Summary of the ADMM performance
abstract type AdmmStatus end
struct MaxStepsReached <: AdmmStatus end
@with_kw struct SmallErrors <: AdmmStatus
  err_primal :: Float64
  err_dual :: Float64
end

@with_kw struct AdmmSummary
  steps_taken :: Int
  termination_status :: AdmmStatus
  total_step_time :: Float64
  total_X_time :: Float64
  total_Y_time :: Float64
  total_Z_time :: Float64
  avg_X_time :: Float64
  avg_Y_time :: Float64
  avg_Z_time :: Float64
  err_hist :: Tuple{VecF64, VecF64}
end

# Parameters during stepation
@with_kw mutable struct AdmmParams
  # Mutable parts
  γ :: VecF64
  zs :: Vector{VecF64}
  λ :: VecF64

  # The stuff you shouldn't mutate
  γdim :: Int = length(γ)
  cinfos :: Vector{Tuple{Int, Int, Int}} # k, kstart, Ckdim
  num_cliques :: Int = length(cinfos)
end

# The cache that we precompute
@with_kw struct AdmmCache
  J :: SpMatF64
  zaff :: SpVecF64
  Hs :: Vector{SpMatF64}

  # The cholesky of the perturbed (D + J*J') + ε*I
  chL :: SpMatF64
  # The diagonal elements that are technically non-zeros
  diagL_hots :: BitArray{1}
end

# Initialize parameters
function initAdmmParams(inst :: QueryInstance, opts :: AdmmSdpOptions)
  init_start_time = time()
  
  # The γ stuff
  γ = zeros(γlength(opts.τ, inst.ffnet))

  # Initialize γlip to be large the upper-bound
  Ws = [M[1:end, 1:end-1] for M in inst.ffnet.Ms]
  γ[end] = sqrt(prod(opnorm(W)^2 for W in Ws))

  # Tdim = sum(inst.ffnet.fdims)
  # γ[1:Tdim] .= 40

  if opts.verbose; @printf("\tinit γlip = %.3f\n", γ[end]) end

  # The vectorized matrices
  cinfos = makeCliqueInfos(opts.τ, inst.ffnet)
  zs = [zeros(Ckdim^2) for (_, _, Ckdim) in cinfos]

  # The λ term
  Zdim = sum(inst.ffnet.edims)
  λ = zeros(Zdim^2)

  # Conclude and return
  init_time = time() - init_start_time
  params = AdmmParams(γ=γ, zs=zs, λ=λ, cinfos=cinfos)
  return params, init_time
end

# Initialize the cache
function initAdmmCache(inst :: QueryInstance, params :: AdmmParams, opts :: AdmmSdpOptions)
  cache_start_time = time()

  # Some useful constants
  ffnet = inst.ffnet
  Zdim = sum(ffnet.edims)
  γdim = params.γdim
  E1 = E(1, ffnet.edims)

  A, B = makeA(ffnet), makeB(ffnet)

  # Compute the affine component first
  zaff = sparse(vec(makeZ(spzeros(γdim), opts.τ, ffnet)))

  # The Jacobian that we gradually fill in
  J = spzeros(Zdim^2, γdim)

  # Computation for index -> (i,j) pairings
  Tdim = sum(ffnet.fdims)
  ijs = [(i,i) for i in 1:Tdim] # Diagonal elements
  ijs = vcat(ijs, [(i,j) for i in 1:(Tdim-1) for j in (i+1):Tdim if j-i <= opts.τ]) # The rest

  for γind in 1:γdim
    # Special case the last term
    if γind == γdim
      # We know that γlip only affects the (1,1) block of Z
      zk = -1 * sparse(vec(E1' * E1))

    # The diagonal terms of T
    elseif 1 <= γind <= Tdim
      Tii = e(γind,Tdim) * e(γind, Tdim)'
      M1ii = makeM1(Tii, A, B, ffnet)
      zk = sparse(vec(M1ii))

    # The cross ij terms
    else
      i, j = ijs[γind]
      δij = e(i,Tdim) - e(j,Tdim)
      Tij = δij * δij'
      M1ij = makeM1(Tij, A, B, ffnet)
      zk = sparse(vec(M1ij))
    end
    J[:,γind] = zk
  end

  # Do the H stuff; the Ec are assumed to be sparse
  Hs = [kron(Ec(k0, Ckd, Zdim), Ec(k0, Ckd, Zdim)) for (_, k0, Ckd) in params.cinfos]
  Hs_hots = [Int.(Hk' * ones(size(Hk)[1])) for Hk in Hs]

  # Do the cholesky stuff
  D = Diagonal(sum(Hs_hots))
  DJJt = D + (opts.ρ * J * J')
  DJJt_reg = Symmetric(DJJt + opts.cholesky_reg_ε * I)

  chol = cholesky(DJJt_reg)
  chL = sparse(chol.L)
  diagL_hots = (diag(chL) .> 1.1 * sqrt(opts.cholesky_reg_ε))

  # TODO: https://discourse.julialang.org/t/cholesky-decomposition-of-low-rank-positive-semidefinite-matrix/70397/3
  # chol = cholesky(DJJt, Val(true), check=false)

  # Prepare to return
  cache_time = time() - cache_start_time
  cache = AdmmCache(J=J, zaff=zaff, Hs=Hs, chL=chL, diagL_hots=diagL_hots)
  return cache, cache_time
end

# Make z vector given the cache
function makez(params :: AdmmParams, cache :: AdmmCache)
  return cache.J * params.γ + cache.zaff
end

function makezksum(params :: AdmmParams, cache :: AdmmCache)
  return sum(cache.Hs[k]' * params.zs[k] for k in 1:params.num_cliques)
end

# Nonnegative projection
function projRplus(γ :: VecF64)
  return max.(γ, 0)
end

# Project a vector onto the NSD cone
function projectNsd(xk :: VecF64)
  xdim = Int(round(sqrt(length(xk)))) # :)
  @assert length(xk) == xdim * xdim
  tmp = Symmetric(reshape(xk, (xdim, xdim)))
  eig = eigen(tmp)
  tmp = Symmetric(eig.vectors * Diagonal(min.(eig.values, 0)) * eig.vectors')
  return tmp[:]
end

# Use a solver to do the stepping X
#   minimize    γlip + (ρ / 2) ||z(γ) - zksum + λ/ρ||^2
#   subject to  γ >= 0
function stepXsolver(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  model = Model(Mosek.Optimizer)
  set_optimizer_attribute(model, "QUIET", true)
  set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", opts.max_solver_X_time)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", opts.solver_X_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", opts.solver_X_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", opts.solver_X_tol)

  # Set up the variables
  num_cliques, γdim = params.num_cliques, params.γdim
  var_γ = @variable(model, [1:γdim])
  # @constraint(model, var_γ[1:γdim] .>= 0)

  # Equality constraints
  z = makez(params, cache)
  zksum = makezksum(params, cache)
  augterm = z - zksum + (params.λ / opts.ρ)

  # The objective
  augnorm = @variable(model)
  @constraint(model, [augnorm; augterm] in SecondOrderCone())

  γnorm = @variable(model)
  @constraint(model, [γnorm; var_γ] in SecondOrderCone())

  γscale = 0.1
  obj = augnorm^2
  # obj = (1/2) * var_γ[end]^2 + (opts.ρ / 2) * augnorm^2
  # obj = (1/2) * var_γ[end]^2 + (γscale * γnorm^2) + (opts.ρ / 2) * augnorm^2

  # obj = var_γ[end]^2 + (γscale * γnorm^2) + sum(norms[k]^2 for k in 1:num_cliques)
  @objective(model, Min, obj)

  # Solve and return
  optimize!(model)
  new_γ = value.(var_γ)
  return new_γ
end

# Y = {z1, ..., zp}
# minimize    (ρ / 2) * ||z(γ) - zksum + λ/ρ||^2
# subject to  each zk <= 0
function stepY(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  zγ = makez(params, cache)
  Hs, zs, num_cliques = cache.Hs, params.zs, params.num_cliques
  zksum = makezksum(params, cache)
  tmps = [Hs[k] * (zγ + zksum + (params.λ/opts.ρ) - (Hs[k]' * zs[k])) for k in 1:num_cliques]
  new_zs = projectNsd.(VecF64.(tmps))
  return new_zs
end

function stepYsolver(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  model = Model(Mosek.Optimizer)
  set_optimizer_attribute(model, "QUIET", true)
  set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", opts.max_solver_X_time)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", opts.solver_X_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", opts.solver_X_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", opts.solver_X_tol)

  # Set up the variables
  num_cliques = params.num_cliques
  var_zs = Vector{Any}()
  for (_, _, Ckdim) in params.cinfos
    var_Zk = @variable(model, [1:Ckdim, 1:Ckdim], Symmetric)
    @SDconstraint(model, var_Zk <= 0)
    push!(var_zs, vec(var_Zk))
  end

  zγ = makez(params, cache)
  zksum = sum(cache.Hs[k]' * var_zs[k] for k in 1:num_cliques)
  augterm = zγ - zksum + (params.λ / opts.ρ)
  augnorm = @variable(model)
  @constraint(model, [augnorm; augterm] in SecondOrderCone())

  # The objective
  obj = augnorm
  @objective(model, Min, obj)

  # Solve and return
  optimize!(model)
  new_zs = [value.(var_zs[k]) for k in 1:num_cliques]
  return new_zs
end

# Z = {λ}
function stepZ(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  zγ = makez(params, cache)
  zksum = makezksum(params, cache)
  new_λ = params.λ + opts.ρ * (zγ - zksum)
  return new_λ
end

# Calculate the primal and dual errors
function stepErrors(prev_params :: AdmmParams, this_params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  zγ = VecF64(makez(this_params, cache))
  num_cliques = this_params.num_cliques
  this_zksum = makezksum(this_params, cache)
  prev_zksum = makezksum(prev_params, cache)

  err_primal = norm(zγ - this_zksum)
  err_dual = norm(opts.ρ * cache.J' * (this_zksum - prev_zksum))
  return err_primal, err_dual
end

# Get the k largest eigenvalues of Z in decending order
function eigvalsZ(k :: Int, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  z = makez(params, cache)
  zdim = Int(round(sqrt(length(z))))
  Z = Symmetric(Matrix(reshape(z, (zdim, zdim))))
  eigsZ = eigvals(Z)
  return sort(eigsZ[end-k+1:end], rev=true)
end

# Go through stuff
function stepAdmm(_params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  step_params = deepcopy(_params)

  steps_taken = 0
  total_step_time, total_X_time, total_Y_time, total_Z_time = 0, 0, 0, 0

  err_primal_hist = VecF64()
  err_dual_hist = VecF64()

  stepXscale = 0.02
  stepYscale = 0.02
  stepZscale = 0.02

  for t in 1:opts.max_steps
    step_start_time = time()
    prev_step_params = deepcopy(step_params)

    # X stuff
    X_start_time = time()
    new_γ = stepXsolver(step_params, cache, opts)
    # step_params.γ = new_γ
    step_params.γ = stepXscale * new_γ + (1 - stepXscale) * step_params.γ
    X_time = time() - X_start_time
    total_X_time += X_time

    # Y stuff
    Y_start_time = time()
    # new_zs = stepY(step_params, cache, opts)
    new_zs = stepYsolver(step_params, cache, opts)
    # step_params.zs = new_zs
    step_params.zs = stepYscale * new_zs + (1 - stepYscale) * step_params.zs
    Y_time = time() - Y_start_time
    total_Y_time += Y_time

    # Z stuff
    Z_start_time = time()
    new_λ = stepZ(step_params, cache, opts)
    # step_params.λ = new_λ
    step_params.λ = stepZscale * new_λ + (1 - stepZscale) * step_params.λ
    Z_time = time() - Z_start_time
    total_Z_time += Z_time

    # Time logistics
    step_time = time() - step_start_time
    total_step_time += step_time
    steps_taken += 1

    # Calculate the error
    err_primal, err_dual = stepErrors(prev_step_params, step_params, cache, opts)
    push!(err_primal_hist, err_primal)
    push!(err_dual_hist, err_dual)
    eigsZ = eigvalsZ(3, step_params, cache, opts)

    # Dump information
    if opts.verbose
      times_str = @sprintf("(X: %.2f, Y: %.2f, Z: %.2f, step: %.2f, total: %.2f)",
                           X_time, Y_time, Z_time, step_time, total_step_time)
      @printf("\tstep[%d/%d] times: %s\n", t, opts.max_steps, times_str)
      @printf("\tγlip: %.3f \terr_primal: %.6f \terr_dual: %.6f\n",
              step_params.γ[end], err_primal, err_dual)
      println("\t\t$(round.(eigsZ', digits=3))")
    end

    # if max(err_primal, err_dual) > 1e3; break end

    if max(err_primal, err_dual) < 1e-3; break end

  end

  summary = AdmmSummary(
    steps_taken = steps_taken,
    termination_status = MaxStepsReached(),
    total_step_time = total_step_time,
    total_X_time = total_X_time,
    total_Y_time = total_Y_time,
    total_Z_time = total_Z_time,
    avg_X_time = total_X_time / steps_taken,
    avg_Y_time = total_Y_time / steps_taken,
    avg_Z_time = total_Z_time / steps_taken,
    err_hist = (err_primal_hist, err_dual_hist))
  return step_params, summary
end

function runQuery(inst :: QueryInstance, opts :: AdmmSdpOptions)
  start_time = time()

  # Initialize some parameters
  init_params, init_time = initAdmmParams(inst, opts)
  cache, cache_time = initAdmmCache(inst, init_params, opts)
  setup_time = init_time + cache_time

  if opts.verbose; @printf("\tcache time: %.3f\n", cache_time) end

  # Do the stepping
  final_params, summary = stepAdmm(init_params, cache, opts)
  total_time = time() - start_time

  if opts.verbose
    @printf("\tsetup time: %.3f \tsolve time: %.3f \ttotal time: %.3f \tvalue: %.3f (%s)\n",
            setup_time, summary.total_step_time, total_time,
            final_params.γ[end], string(summary.termination_status))
  end

  return SolutionOutput(
    objective_value = final_params.γ[end],
    values = final_params,
    summary = summary,
    termination_status = summary.termination_status,
    total_time = total_time,
    setup_time = setup_time,
    solve_time = summary.total_step_time)
end

