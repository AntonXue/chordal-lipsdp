using Parameters
using LinearAlgebra
using SparseArrays
using JuMP
using MosekTools
using Printf

# Options for ADMM
@with_kw struct AdmmSdpOptions
  τ::Int = 0
  max_steps::Int = 10000

  ρ_init::Float64 = 2^3
  ρ_scale::Float64 = 2
  ρ_rel_gap::Float64 = 5
  ρ_max::Float64 = 2^4
  ρ_min::Float64 = 2^(-4)
  @assert ρ_min <= ρ_init <= ρ_max

  nsd_tol::Float64 = 1e-3
  stop_at_first_nsd::Bool = true
  
  max_solver_X_time::Float64 = 60.0
  solver_X_tol::Float64 = 1e-3

  max_solver_Y_time::Float64 = 60.0
  solver_Y_tol::Float64 = 1e-3

  verbose::Bool = false
  verbose_every_t::Int = 1
  check_sat_every_t::Int = 1
end

# Summary of the ADMM performance
abstract type AdmmStatus end
struct MaxStepsReached <: AdmmStatus end
@with_kw struct SmallErrors <: AdmmStatus
  err_primal::Float64
  err_dual::Float64
end

@with_kw struct AdmmSummary
  steps_taken::Int
  termination_status::AdmmStatus
  total_step_time::Float64
  total_X_time::Float64
  total_Y_time::Float64
  total_Z_time::Float64
  avg_X_time::Float64
  avg_Y_time::Float64
  avg_Z_time::Float64
  err_primal_hist::VecF64
  err_dual_hist::VecF64
  err_aff_hist::VecF64
  λmax_hist::VecF64
end

# Parameters during stepation
@with_kw mutable struct AdmmParams
  # Mutable parts
  ω::VecF64
  vs::Vector{VecF64}
  γ::VecF64
  zs::Vector{VecF64}
  μ::VecF64
  λs::Vector{VecF64}
  ρ::Float64

  # The stuff you shouldn't mutate
  γdim::Int = length(γ)
  Zdim::Int
  cinfos::Vector{Tuple{Int, Int, Int}} # k, kstart, Ckdim
  num_cliques::Int = length(cinfos)
end

# The cache that we precompute
@with_kw struct AdmmCache
  J::SpMatF64
  zaff::SpVecF64
  zaff_f64 = VecF64(zaff)
  Hs::Vector{SpMatF64}

  chol
  spL = sparse(chol.L)
  spL_r = spL[1:chol.rank, 1:chol.rank]
  spLt_r = sparse(spL_r')

end

# Initialize parameters
function initAdmmParams(inst::QueryInstance, opts::AdmmSdpOptions)
  init_start_time = time()
  
  # The γ stuff
  Zdim = sum(inst.ffnet.edims)
  γ = sqrt(Zdim) * ones(γlength(opts.τ, inst.ffnet))
  ω = γ

  # Attempt a smart initialization γlip to be large the upper-bound
  # Ws = [M[:, 1:end-1] for M in inst.ffnet.Ms]
  # γ[end] = (prod(opnorm(W) for W in Ws)) / 2

  if opts.verbose; @printf("\tinit γlip = %.3f\n", γ[end]) end

  # The vectorized matrices
  cinfos = makeCliqueInfos(opts.τ, inst.ffnet)

  # The Z we would have given the γ above; use this to guess initial values
  Z = makeZ(γ, opts.τ, inst.ffnet)

  # The vs get wiped out in the first X step anyways
  vs = [zeros(Ckdim^2) for (_, _, Ckdim) in cinfos]
  zs = [zeros(Ckdim^2) for (_, _, Ckdim) in cinfos]
  # Ecs = [Ec(kstart, Ckdim, Zdim) for (_, kstart, Ckdim) in cinfos]
  # zs = [vec(Eck * Z * Eck') for Eck in Ecs]

  # Guess some values of λ
  μ = zeros(length(γ))
  λs = [zeros(Ckdim^2) for (_, _, Ckdim) in cinfos]
  # Λs = [-sqrt(Zdim) * I(Ckdim) for (_, _, Ckdim) in cinfos]
  # λs = [VecF64(vec(Λk)) for Λk in Λs]

  ρ = opts.ρ_init

  # Conclude and return
  init_time = time() - init_start_time
  params = AdmmParams(γ=γ, vs=vs, ω=ω, zs=zs, μ=μ, λs=λs, ρ=ρ, Zdim=Zdim, cinfos=cinfos)
  return params, init_time
end

# Initialize the cache
function initAdmmCache(inst::QueryInstance, params::AdmmParams, opts::AdmmSdpOptions)
  cache_start_time = time()

  # Some useful constants
  ffnet = inst.ffnet
  Zdim = params.Zdim
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
  # Hs_hots = [Int.(Hk' * ones(size(Hk)[1])) for Hk in Hs]
  
  D = sum(Hk' * Hk for Hk in Hs)
  DJJt = D + J * J'
  chol_start_time = time()
  chol = cholesky(Symmetric(Matrix(DJJt)), Val(true), check=false)
  @printf("cholesky time: %.3f\n", time() - chol_start_time)

  # Prepare to return
  cache_time = time() - cache_start_time
  cache = AdmmCache(J=J, zaff=zaff, Hs=Hs, chol=chol)
  return cache, cache_time
end

# Make z vector given the cache
function makez(params::AdmmParams, cache::AdmmCache)
  return cache.J * params.γ + cache.zaff
end

# Nonnegative projection
function projectRplus(γ::VecF64)
  return max.(γ, 0)
end

# Project a vector onto the NSD cone
function projectNsd(xk::VecF64)
  xdim = Int(round(sqrt(length(xk)))) # :)
  @assert length(xk) == xdim * xdim
  tmp = Symmetric(reshape(xk, (xdim, xdim)))
  eig = eigen(tmp)
  tmp = Symmetric(eig.vectors * Diagonal(min.(eig.values, 0)) * eig.vectors')
  return tmp[:]
end

# Step X
#   minimize  ωlip + (ρ/2) sum ||zk - vk + λk/ρ||^2 + (ρ/2) ||γ - ω + μ/ρ||^2
#   subj to   z(ω) = sum Hk' * vk
function stepX(params::AdmmParams, cache::AdmmCache, opts::AdmmSdpOptions)
  Hzλsum = sum(cache.Hs[k]' * (params.zs[k] + (params.λs[k] / params.ρ)) for k in 1:params.num_cliques)
  u1 = VecF64(Hzλsum - cache.zaff)
  u2 = VecF64(params.γ + (1 / params.ρ) * (params.μ - e(params.γdim, params.γdim)))

  # Solve for x via P L Lt Pt x = J * u2 - u1
  # First solve P L y = J * u2 - u1
  chol = cache.chol
  rhs = cache.J * u2 - u1
  rhs_r = (rhs[chol.p])[1:chol.rank]
  y = cache.spL_r \ rhs_r
  @assert length(y) == chol.rank

  # Then solve Lt Pt x = y
  Ptx_r = cache.spLt_r \ y
  Ptx = [Ptx_r; zeros(length(rhs) - chol.rank + 1)]
  x = Ptx[invperm(chol.p)]

  # Now can solve for ω and vs
  new_ω = u2 - cache.J' * x
  new_vs = [params.zs[k] + (params.λs[k] / params.ρ) + cache.Hs[k] * x for k in 1:params.num_cliques]
  return new_ω, new_vs
end

# Use a solver to do the stepping
function stepXsolver(params::AdmmParams, cache::AdmmCache, opts::AdmmSdpOptions)
  model = Model(Mosek.Optimizer)
  set_optimizer_attribute(model, "QUIET", true)
  set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", opts.max_solver_X_time)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", opts.solver_X_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", opts.solver_X_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", opts.solver_X_tol)

  # Set up the variables
  ωdim = params.γdim
  var_ω = @variable(model, [1:ωdim])

  var_vs = Vector{Any}()
  for (k, _, Ckdim) in params.cinfos
    var_vk = @variable(model, [1:Ckdim^2])
    push!(var_vs, var_vk)
  end

  # Equality constraints
  z = cache.J * var_ω + cache.zaff
  vksum = sum(cache.Hs[k]' * var_vs[k] for k in 1:params.num_cliques)
  @constraint(model, z .== vksum)

  # Norm stuff, with the λ terms first, then the μ term
  var_norms = @variable(model, [1:params.num_cliques+1])
  for k in 1:params.num_cliques
    λtermk = params.zs[k] - var_vs[k] + (params.λs[k] / params.ρ)
    @constraint(model, [var_norms[k]; λtermk] in SecondOrderCone())
  end

  # The μ penalty
  μterm = params.γ - var_ω + (params.μ / params.ρ)
  @constraint(model, [var_norms[end]; μterm] in SecondOrderCone())
  
  # Form the objective
  obj = var_ω[end] + (params.ρ / 2) * sum(n^2 for n in var_norms)
  @objective(model, Min, obj)

  # Solve and return
  optimize!(model)
  new_ω = value.(var_ω)
  new_vs = [value.(var_vk) for var_vk in var_vs]
  return new_ω, new_vs
end

# Y = {z1, ..., zp}
# minimize    (ρ/2) sum ||zk - vk + λk/ρ||^2 + (ρ/2) ||γ - ω + μ/ρ||^2
# subject to  ω >= 0 and each zk <= 0
function stepY(params::AdmmParams, cache::AdmmCache, opts::AdmmSdpOptions)
  tmp_γ = params.ω - (params.μ / params.ρ)
  tmp_zs = [params.vs[k] - (params.λs[k] / params.ρ) for k in 1:params.num_cliques]
  new_γ = projectRplus(tmp_γ)
  new_zs = projectNsd.(tmp_zs)
  return new_γ, new_zs
end

function stepYsolver(params::AdmmParams, cache::AdmmCache, opts::AdmmSdpOptions)
  model = Model(Mosek.Optimizer)
  set_optimizer_attribute(model, "QUIET", true)
  set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", opts.max_solver_Y_time)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", opts.solver_Y_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", opts.solver_Y_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", opts.solver_Y_tol)

  # Set up the variables
  var_γ = @variable(model, [1:params.γdim])
  @constraint(model, var_γ[1:params.γdim] .>= 0)

  var_zs = Vector{Any}()
  for (_, _, Ckdim) in params.cinfos
    var_Zk = @variable(model, [1:Ckdim, 1:Ckdim], Symmetric)
    @constraint(model, -var_Zk in PSDCone())
    push!(var_zs, vec(var_Zk))
  end

  # The norm terms with the λ terms first, and the μ term last
  var_norms = @variable(model, [1:params.num_cliques+1])
  for k in 1:params.num_cliques
    λtermk = var_zs[k] - params.vs[k] + (params.λs[k] / params.ρ)
    @constraint(model, [var_norms[k]; λtermk] in SecondOrderCone())
  end

  # The μ penalty term
  μterm = var_γ - params.ω + (params.μ / params.ρ)
  @constraint(model, [var_norms[end]; μterm] in SecondOrderCone())

  # The objective
  obj = sum(n^2 for n in var_norms)
  @objective(model, Min, obj)

  # Solve and return
  optimize!(model)
  new_γ = value.(var_γ)
  new_zs = [value.(var_zk) for var_zk in var_zs]
  return new_γ, new_zs
end

# Z = {μ, λ}
function stepZ(params::AdmmParams, cache::AdmmCache, opts::AdmmSdpOptions)
  new_μ = params.μ + params.ρ * (params.γ - params.ω)
  new_λs = [params.λs[k] + params.ρ * (params.zs[k] - params.vs[k]) for k in 1:params.num_cliques]
  return new_μ, new_λs
end

# Calculate the primal and dual errors
function stepErrors(prev_params::AdmmParams, this_params::AdmmParams, cache::AdmmCache, opts::AdmmSdpOptions)
  num_cliques = this_params.num_cliques
  this_ω = this_params.ω
  this_vs = this_params.vs
  this_γ = this_params.γ
  this_zs = this_params.zs
  this_μ = this_params.μ
  this_λs = this_params.λs
  this_ρ = this_params.ρ
  prev_γ = prev_params.γ
  prev_zs = prev_params.zs
  prev_μ = prev_params.μ
  prev_λs = prev_params.λs

  err_primal1 = sum(norm(this_zs[k] - this_vs[k])^2 for k in 1:num_cliques)
  err_primal2 = norm(this_γ - this_ω)^2
  err_primal = sqrt(err_primal1 + err_primal2)

  err_dual1 = sum(norm(this_zs[k] - prev_zs[k])^2 for k in 1:num_cliques)
  err_dual2 = norm(this_γ - prev_γ)^2
  err_dual = this_params.ρ * sqrt(err_dual1 + err_dual2)

  # err_dual1 = sum(norm(this_λs[k] - prev_λs[k])^2 for k in 1:num_cliques)
  # err_dual2 = norm(this_μ - prev_μ)^2
  # err_dual = this_params.ρ * sqrt(err_dual1 + err_dual2)

  err_aff1 = makez(this_params, cache)
  err_aff2 = sum(cache.Hs[k]' * this_zs[k] for k in 1:num_cliques) # Compare with zk instead of vk
  err_aff = norm(err_aff1 - err_aff2)

  return err_primal, err_dual, err_aff
end

# Get the k largest eigenvalues of Z in decending order
function eigvalsZ(k::Int, params::AdmmParams, cache::AdmmCache, opts::AdmmSdpOptions)
  z = cache.J * params.γ + cache.zaff
  zdim = Int(round(sqrt(length(z))))
  Z = Symmetric(Matrix(reshape(z, (zdim, zdim))))
  eigsZ = eigvals(Z)
  return sort(eigsZ[end-k+1:end], rev=true)
end

# Go through stuff
function stepAdmm(_params::AdmmParams, cache::AdmmCache, opts::AdmmSdpOptions)
  step_params = deepcopy(_params)

  num_cliques = step_params.num_cliques
  steps_taken = 0
  total_step_time, total_X_time, total_Y_time, total_Z_time = 0, 0, 0, 0

  err_primal_hist = VecF64()
  err_dual_hist = VecF64()
  err_aff_hist = VecF64()
  λmax_hist = VecF64()

  αx, αy, αz = 1.0, 1.0, 1.0

  for t in 1:opts.max_steps
    step_start_time = time()
    prev_step_params = deepcopy(step_params)

    # X stuff
    X_start_time = time()
    new_ω, new_vs = stepX(step_params, cache, opts)
    # new_ω, new_vs = stepXsolver(step_params, cache, opts)
    step_params.ω = αx * new_ω + (1 - αx) * step_params.ω
    step_params.vs = αx * new_vs + (1 - αx) * step_params.vs
    X_time = time() - X_start_time
    total_X_time += X_time

    # Y stuff
    Y_start_time = time()
    new_γ, new_zs = stepY(step_params, cache, opts)
    # new_γ, new_zs = stepYsolver(step_params, cache, opts)
    step_params.γ = αy * new_γ + (1 - αy) * step_params.γ
    step_params.zs = αy * new_zs + (1 - αy) * step_params.zs
    Y_time = time() - Y_start_time
    total_Y_time += Y_time

    # Z stuff
    Z_start_time = time()
    new_μ, new_λs = stepZ(step_params, cache, opts)
    step_params.μ = αz * new_μ + (1 - αz) * step_params.μ
    step_params.λs = αz * new_λs + (1 - αz) * step_params.λs
    Z_time = time() - Z_start_time
    total_Z_time += Z_time

    # Time logistics
    step_time = time() - step_start_time
    total_step_time += step_time
    steps_taken += 1

    # Calculate the error
    err_primal, err_dual, err_aff = stepErrors(prev_step_params, step_params, cache, opts)
    push!(err_primal_hist, err_primal)
    push!(err_dual_hist, err_dual)
    push!(err_aff_hist, err_aff)
    
    if step_params.ρ > opts.ρ_max
      step_params.ρ = opts.ρ_max
    end

    if step_params.ρ < opts.ρ_min
      step_params.ρ = opts.ρ_min
    end

    # Adaptively adjust the penalty parameter ρ
    if (step_params.ρ < opts.ρ_max) && (err_primal > err_dual * opts.ρ_rel_gap)
      step_params.ρ *= opts.ρ_scale
    end

    if (step_params.ρ > opts.ρ_min) && (err_dual > err_primal * opts.ρ_rel_gap)
      step_params.ρ /= opts.ρ_scale
    end

    # Dump information
    if opts.verbose && mod(t, opts.verbose_every_t) == 0
      times_str = @sprintf("(X: %.2f, Y: %.2f, Z: %.2f, step: %.2f, total: %.2f)",
                           X_time, Y_time, Z_time, step_time, total_step_time)
      @printf("\tstep[%d/%d] times: %s\n", t, opts.max_steps, times_str)
      @printf("\terr_primal: %.6f \terr_dual: %.6f \terr_aff: %.6f\n",
              err_primal, err_dual, err_aff)
      @printf("\tγlip: %.4f \tnew_ρ: %.4f\n",
              step_params.γ[end], step_params.ρ)
    end

    if max(err_primal, err_dual) < 1e-3; break end
    if max(err_primal, err_dual) > 1e8; break end

    # This termination check is based on ω because it is actually projected
    if mod(t, opts.check_sat_every_t) == 0
      z = VecF64(cache.J * step_params.ω + cache.zaff)
      Z = Symmetric(reshape(z, (step_params.Zdim, step_params.Zdim)))
      λmaxZ = eigmax(Z)
      push!(λmax_hist, λmaxZ)
      if opts.verbose && mod(t, opts.verbose_every_t) == 0
        @printf("\t\tλmax Z(γ): %.4f\n", λmaxZ)
      end
      # if opts.stop_at_first_nsd && λmaxZ < opts.nsd_tol; break end
    end

    if opts.verbose && mod(t, opts.verbose_every_t) == 0
      println("\n")
    end
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
    err_primal_hist = err_primal_hist,
    err_dual_hist = err_dual_hist,
    err_aff_hist = err_aff_hist,
    λmax_hist = λmax_hist)
  return step_params, summary
end

function runQuery(inst::QueryInstance, opts::AdmmSdpOptions)
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

  return QuerySolution(
    objective_value = final_params.γ[end],
    values = final_params,
    summary = summary,
    termination_status = summary.termination_status,
    total_time = total_time,
    setup_time = setup_time,
    solve_time = summary.total_step_time)
end

