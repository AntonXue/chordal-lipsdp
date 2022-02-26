using Parameters
using LinearAlgebra
using SparseArrays
using JuMP
using MosekTools
using Printf

# Options for ADMM
@with_kw struct AdmmSdpOptions
  τ :: Int = 0
  max_steps :: Int = 200

  ρ_init :: Float64 = 1.0
  ρ_scale :: Float64 = 2
  ρ_rel_gap :: Float64 = 10
  ρ_max :: Float64 = 2^12
  ρ_min :: Float64 = 2^(-12)

  nsd_tol :: Float64 = 1e-3
  stop_at_first_nsd :: Bool = true
  
  max_solver_X_time :: Float64 = 60.0
  solver_X_tol :: Float64 = 1e-3

  max_solver_Y_time :: Float64 = 60.0
  solver_Y_tol :: Float64 = 1e-3
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
  err_primal_hist :: VecF64
  err_dual_hist :: VecF64
  err_rel_hist :: VecF64
  err_aff_hist :: VecF64
  λmax_hist :: VecF64
end

# Parameters during stepation
@with_kw mutable struct AdmmParams
  # Mutable parts
  γ :: VecF64
  vs :: Vector{VecF64}
  zs :: Vector{VecF64}
  λs :: Vector{VecF64}
  ρ :: Float64

  # The stuff you shouldn't mutate
  γdim :: Int = length(γ)
  cinfos :: Vector{Tuple{Int, Int, Int}} # k, kstart, Ckdim
  num_cliques :: Int = length(cinfos)
end

# The cache that we precompute
@with_kw struct AdmmCache
  J :: SpMatF64
  zaff :: SpVecF64
  zaff_f64 = VecF64(zaff)
  Hs :: Vector{SpMatF64}

  #=
  chol
  D
  pinvD
  Jt_pinvD
  pinvD_J
  Jt_pinvD_J
  inv_Jt_pinvD_J
  =#

  HJ
  qrft
  qrft_rank = rank(qrft.R)
  Qt1 = sparse(Matrix(qrft.Q)')
  Rt1 = qrft.R'[1:qrft_rank, :]
end

# Initialize parameters
function initAdmmParams(inst :: QueryInstance, opts :: AdmmSdpOptions)
  init_start_time = time()
  
  # The γ stuff
  γ = zeros(γlength(opts.τ, inst.ffnet))

  # Attempt a smart initialization γlip to be large the upper-bound
  # Ws = [M[:, 1:end-1] for M in inst.ffnet.Ms]
  # γ[end] = sqrt(prod(opnorm(W) for W in Ws))
  # Tdim = sum(inst.ffnet.fdims)
  # γ[1:Tdim] .= 40

  if opts.verbose; @printf("\tinit γlip = %.3f\n", γ[end]) end

  # The vectorized matrices
  cinfos = makeCliqueInfos(opts.τ, inst.ffnet)
  vs = [zeros(Ckdim^2) for (_, _, Ckdim) in cinfos]
  zs = [zeros(Ckdim^2) for (_, _, Ckdim) in cinfos]
  λs = [zeros(Ckdim^2) for (_, _, Ckdim) in cinfos]

  ρ = opts.ρ_init

  # Conclude and return
  init_time = time() - init_start_time
  params = AdmmParams(γ=γ, vs=vs, zs=zs, λs=λs, ρ=ρ, cinfos=cinfos)
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
  # Hs_hots = [Int.(Hk' * ones(size(Hk)[1])) for Hk in Hs]

  # Cache information for the KKT system
  # [D -J; -Jt 0]
  # Set up the KKT matrix of [D At; A 0], where At = -J
  D = Diagonal(diag(sum(Hk' * Hk for Hk in Hs)))
  pinvD = Diagonal(pinv(D))
  
  Jt_pinvD = J' * pinvD
  pinvD_J = pinvD * J
  Jt_pinvD_J = J' * pinvD * J
  inv_Jt_pinvD_J = inv(Matrix(Jt_pinvD_J))

  chol_start_time = time()
  @printf("starting chol\n")
  chol = cholesky(Jt_pinvD_J)
  @printf("chol took time: %.3f\n", time() - chol_start_time)

  HJt = [vcat([Hk for Hk in Hs]...); -J']
  qr_start_time = time()
  @printf("starting qr\n")
  qrft = qr(HJt)
  @printf("qr took time: %.3f\n", time() - qr_start_time)


  HJ = sparse(HJt')
  qrft_rank = rank(qrft.R)

  # Prepare to return
  cache_time = time() - cache_start_time
  cache = AdmmCache(J=J, zaff=zaff, Hs=Hs, HJ=HJ, qrft=qrft)

  # cache_time = time() - cache_start_time
  # cache = AdmmCache(J=J, zaff=zaff, Hs=Hs, chol=0, pinvD=0, neg_Jt_pinvD=0, neg_pinvD_J=0, inv_Jt_pinvD_J=0)
  return cache, cache_time
end

# Make z vector given the cache
function makez(params :: AdmmParams, cache :: AdmmCache)
  return cache.J * params.γ + cache.zaff
end

# Nonnegative projection
function projectRplus(γ :: VecF64)
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

function stepX(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  γdim, num_cliques = params.γdim, params.num_cliques
  # First solve the KKT system
  # Rhs of equation (3.27) in Zheng 2019
  u1 = -cache.zaff + sum(cache.Hs[k]' * (params.zs[k] + (params.λs[k] / params.ρ)) for k in 1:num_cliques)
  u2 = (-1 / params.ρ) * e(γdim, γdim) # For us, -b = e(γdim, γdim)

  # First solve for γ
  # L = sparse(cache.chol.L)
  # w = L \ (cache.neg_Jt_pinvD * u1 - u2)
  # γ_opt = L' \ w
  
  γ_opt = cache.inv_Jt_pinvD_J * (u2 + cache.Jt_pinvD * u1)
  γ_opt = projectRplus(γ_opt) # The placement of this seems to matter

  # println("\t\tlargest some values of γopt")
  # println("\t\t$(sort(abs.(round.(γ_opt, digits=4)), rev=true)[1:3])")

  # Now can solve for x
  x_opt = cache.pinvD_J * γ_opt - cache.pinvD * u1

  # Update vs
  new_γ = γ_opt
  new_vs = [params.zs[k] + (params.λs[k] / params.ρ) + cache.Hs[k] * x_opt for k in 1:num_cliques]
  
  return new_γ, new_vs
end

# Use a solver to do the stepping X
#   minimize    γlip^2 + (ρ / 2) sum ||zk - vk + λk/ρ||^2
#   subject to  γ >= 0 and z(γ) = sum Hk' * vk
function stepXsolver(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  model = Model(Mosek.Optimizer)
  set_optimizer_attribute(model, "QUIET", true)
  set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", opts.max_solver_X_time)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", opts.solver_X_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", opts.solver_X_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", opts.solver_X_tol)

  # Set up the variables
  num_cliques = params.num_cliques
  γdim = params.γdim
  var_γ = @variable(model, [1:γdim])
  @constraint(model, var_γ[1:γdim] .>= 0)

  var_vs = Vector{Any}()
  for (k, _, Ckdim) in params.cinfos
    var_vk = @variable(model, [1:Ckdim^2])
    push!(var_vs, var_vk)
  end

  # Equality constraints
  # z = makez(params, cache)
  z = cache.J * var_γ + cache.zaff
  vksum = sum(cache.Hs[k]' * var_vs[k] for k in 1:num_cliques)
  @constraint(model, z .== vksum)

  # The terms of the penalty
  augnorms = @variable(model, [1:num_cliques])
  for k in 1:num_cliques
    augk = params.zs[k] - var_vs[k] + (params.λs[k] / params.ρ)
    @constraint(model, [augnorms[k]; augk] in SecondOrderCone())
  end

  obj = var_γ[end] + (params.ρ / 2) * sum(augnorms[k]^2 for k in 1:num_cliques)
  @objective(model, Min, obj)

  # Solve and return
  optimize!(model)
  new_γ = value.(var_γ)
  new_vs = [value.(var_vk) for var_vk in var_vs]
  return new_γ, new_vs
end

# Another version of the step solver
function stepXsolver2(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  model = Model(Mosek.Optimizer)
  set_optimizer_attribute(model, "QUIET", true)
  set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", opts.max_solver_X_time)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", opts.solver_X_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", opts.solver_X_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", opts.solver_X_tol)

  # Set up the variables
  num_cliques = params.num_cliques
  γdim = params.γdim
  var_γ = @variable(model, [1:γdim])
  @constraint(model, var_γ[1:γdim] .>= 0)

  xdim = size(cache.pinvD)[1]
  var_x = @variable(model, [1:xdim])

  u1 = -cache.zaff + sum(cache.Hs[k]' * (params.zs[k] + (params.λs[k] / params.ρ)) for k in 1:num_cliques)
  u2 = (-1 / params.ρ) * e(γdim, γdim) # For us, -b = e(γdim, γdim)

  lhs1 = -cache.D * var_x + cache.J * var_γ
  lhs2 = cache.J' * var_x

  diff = [lhs1; lhs2] - [u1; u2]

  diffnorm = @variable(model)
  @constraint(model, [diffnorm; diff] in SecondOrderCone())
  @objective(model, Min, diffnorm)

  # The affine stuff
  # @constraint(model, cache.Jt_pinvD_J * var_γ .== u2 + cache.Jt_pinvD * u1)
  # @constraint(model, cache.D * var_x .== cache.J * var_γ - u1)
  # @constraint(model, -cache.D * var_x + cache.J * var_γ .== u1)
  # @constraint(model, cache.J' * var_x .== u2)

  # @objective(model, Min, var_γ[end])

  # Solve and return
  optimize!(model)
  x_val = value.(var_x)
  new_vs = [params.zs[k] + (params.λs[k] / params.ρ) + cache.Hs[k] * x_val for k in 1:num_cliques]
  new_γ = value.(var_γ)
  new_γ = projectRplus(new_γ)

  println("\tstepXsolver2 term status: $(solution_summary(model).termination_status)")

  return new_γ, new_vs
end

# Run a few rounds of projected gradient descent
function stepXiterative(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  num_cliques = params.num_cliques
  γdim = params.γdim

  # The componetns of the iterative variable x = [v1; ...; vp; γ]
  xdims = [length.(params.vs); γdim]
  xdim = sum(xdims)

  # The initial values
  v0s = [params.zs[k] + (params.λs[k] / params.ρ) for k in 1:num_cliques]
  γ0 = params.γ
  xt = [vcat(v0s...); γ0]

  α = 0.5
  # Begin the iterations
  for t = 1:3
    xparts = splice(xt, xdims)
    @assert length(xparts) == num_cliques + 1

    # First the gradient step
    ∇vs = [-params.ρ * (params.zs[k] - xparts[k] + (params.λs[k] / params.ρ)) for k in 1:num_cliques]
    ∇γ = e(γdim, γdim)
    ∇f = [vcat(∇vs...); ∇γ]

    # Take a step
    yt1 = xt - α * ∇f

    # Do the projection for z(γ) = sum Hk' vk
    


    # Do the projection for γ >= 0

    xt = yt1
  end

  return v0s, γ0, xt
end

# Y = {z1, ..., zp}
# minimize    (ρ / 2) sum ||zk - vk + λk/ρ||^2
# subject to  each zk <= 0
function stepY(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  tmps = [params.vs[k] - (params.λs[k] / params.ρ) for k in 1:params.num_cliques]
  new_zs = projectNsd.(tmps)
  return new_zs
end

function stepYsolver(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  model = Model(Mosek.Optimizer)
  set_optimizer_attribute(model, "QUIET", true)
  set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", opts.max_solver_Y_time)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", opts.solver_Y_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", opts.solver_Y_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", opts.solver_Y_tol)

  # Set up the variables
  num_cliques = params.num_cliques
  var_Zs = Vector{Any}()
  for (_, _, Ckdim) in params.cinfos
    var_Zk = @variable(model, [1:Ckdim, 1:Ckdim], Symmetric)
    @SDconstraint(model, var_Zk <= 0)
    push!(var_Zs, var_Zk)
  end

  augnorms = @variable(model, [1:num_cliques])
  for k in 1:num_cliques
    augk = vec(var_Zs[k]) - params.vs[k] + (params.λs[k] / params.ρ)
    @constraint(model, [augnorms[k]; augk] in SecondOrderCone())
  end

  obj = sum(augnorms[k]^2 for k in 1:num_cliques)
  @objective(model, Min, obj)

  # Solve and return
  optimize!(model)
  new_zs = [vec(value.(var_Zk)) for var_Zk in var_Zs]
  return new_zs
end

# Z = {λ}
function stepZ(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  new_λs = [params.λs[k] + params.ρ * (params.zs[k] - params.vs[k]) for k in 1:params.num_cliques]
  return new_λs
end

# Calculate the primal and dual errors
function stepErrors(prev_params :: AdmmParams, this_params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  num_cliques = this_params.num_cliques
  this_zs = this_params.zs
  this_vs = this_params.vs
  this_λs = this_params.λs
  prev_zs = prev_params.zs

  err_primal = sqrt(sum(norm(this_zs[k] - this_vs[k])^2 for k in 1:num_cliques))

  err_dual1 = sqrt(sum(norm(this_zs[k] - prev_zs[k])^2 for k in 1:num_cliques))
  err_dual2 = sqrt(sum(norm(this_λs[k])^2 for k in 1:num_cliques))
  err_dual = this_params.ρ * err_dual1 / err_dual2

  err_rel1 = sqrt(sum(norm(this_zs[k])^2 for k in 1:num_cliques))
  err_rel2 = sqrt(sum(norm(this_vs[k])^2 for k in 1:num_cliques))
  err_rel = err_primal / max(err_rel1, err_rel2)

  err_aff1 = makez(this_params, cache)
  err_aff2 = sum(cache.Hs[k]' * this_zs[k] for k in 1:num_cliques) # Compare with zk instead of vk
  err_aff = norm(err_aff1 - err_aff2)

  return err_primal, err_dual, err_rel, err_aff
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

  num_cliques = step_params.num_cliques
  steps_taken = 0
  total_step_time, total_X_time, total_Y_time, total_Z_time = 0, 0, 0, 0

  err_primal_hist = VecF64()
  err_dual_hist = VecF64()
  err_rel_hist = VecF64()
  err_aff_hist = VecF64()
  λmax_hist = VecF64()

  αx, αy, αz = 1.0, 1.0, 1.0

  for t in 1:opts.max_steps
    step_start_time = time()
    prev_step_params = deepcopy(step_params)

    # X stuff
    X_start_time = time()
    # new_γ, new_vs = stepX(step_params, cache, opts)
    new_γ, new_vs = stepXsolver(step_params, cache, opts)
    # new_γ, new_vs = stepXsolver2(step_params, cache, opts)
    step_params.γ = αx * new_γ + (1 - αx) * step_params.γ
    step_params.vs = [αx * new_vs[k] + (1 - αx) * step_params.vs[k] for k in 1:num_cliques] 
    X_time = time() - X_start_time
    total_X_time += X_time

    # Y stuff
    Y_start_time = time()
    new_zs = stepY(step_params, cache, opts)
    # new_zs = stepYsolver(step_params, cache, opts)
    step_params.zs = [αy * new_zs[k] + (1 - αy) * step_params.zs[k] for k in 1:num_cliques]
    Y_time = time() - Y_start_time
    total_Y_time += Y_time

    # Z stuff
    Z_start_time = time()
    new_λs = stepZ(step_params, cache, opts)
    step_params.λs = [αz * new_λs[k] + (1 - αz) * step_params.λs[k] for k in 1:num_cliques]
    Z_time = time() - Z_start_time
    total_Z_time += Z_time

    # Time logistics
    step_time = time() - step_start_time
    total_step_time += step_time
    steps_taken += 1

    # Calculate the error
    err_primal, err_dual, err_rel, err_aff = stepErrors(prev_step_params, step_params, cache, opts)
    push!(err_primal_hist, err_primal)
    push!(err_dual_hist, err_dual)
    push!(err_rel_hist, err_rel)
    push!(err_aff_hist, err_aff)
    
    # Adaptively adjust the penalty parameter ρ
    if (step_params.ρ < opts.ρ_max) && (err_primal > err_dual * opts.ρ_rel_gap)
      step_params.ρ *= opts.ρ_scale
    end

    if (step_params.ρ > opts.ρ_min) && (err_dual > err_primal * opts.ρ_rel_gap)
      step_params.ρ /= opts.ρ_scale
    end


    eigsZ = eigvalsZ(3, step_params, cache, opts)
    push!(λmax_hist, eigsZ[1])

    # Dump information
    if opts.verbose
      times_str = @sprintf("(X: %.2f, Y: %.2f, Z: %.2f, step: %.2f, total: %.2f)",
                           X_time, Y_time, Z_time, step_time, total_step_time)
      @printf("\tstep[%d/%d] times: %s\n", t, opts.max_steps, times_str)
      @printf("\terr_primal: %.6f \terr_dual: %.6f \terr_rel: %.6f \terr_aff: %.6f\n",
              err_primal, err_dual, err_rel, err_aff)
      @printf("\tγlip: %.4f \tnew_ρ: %.4f\n",
              step_params.γ[end], step_params.ρ)
      println("\t\t$(round.(eigsZ', digits=3))")
    end

    # if max(err_primal, err_dual) > 1e3; break end

    if max(err_primal, err_dual) < 1e-3; break end

    if max(err_primal, err_dual) > 1e8; break end

    if opts.stop_at_first_nsd && eigsZ[1] < opts.nsd_tol; break end


    @printf("\n")

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
    err_rel_hist = err_rel_hist,
    err_aff_hist = err_aff_hist,
    λmax_hist = λmax_hist)
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

