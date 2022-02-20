using Parameters
using LinearAlgebra
using SparseArrays
using Printf

@with_kw struct AdmmSdpOptions
  τ :: Int = 0
  lagρ :: Float64 = 1.0
  max_iters :: Int = 200
  
  solver_X_max_time :: Float64 = 60.0
  solver_X_tol :: Float64 = 1e-4
  cholesky_reg_ε :: Float64 = 1e-2
  verbose :: Bool = false
end

@with_kw mutable struct AdmmParams
  # Mutable parts
  γ :: VecF64
  vs :: Vector{VecF64}
  zs :: Vector{VecF64}
  λs :: Vector{VecF64}

  # The stuff you shouldn't mutate
  γdim :: Int = length(γ)
  cinfos :: Vector{Tuple{Int, Int, Int}} # k, kstart, Ckdim
end

@with_kw struct AdmmCache
  J :: SpMatF64
  zaff :: SpVecF64
  Hs :: Vector{SpMatF64}

  # The cholesky of the perturbed (D + J*J') + ε*I
  chL :: SpMatF64
  # The diagonal elements that are technically non-zeros
  diagL_hots :: BitArray{1}
end


function initAdmmParams(inst :: QueryInstance, opts :: AdmmSdpOptions)
  init_start_time = time()
  
  # The γ stuff
  γ = zeros(γlength(opts.τ, inst.ffnet)) # Need to store ρ at the end

  # The vectorized matrices
  cinfos = makeCliqueInfos(opts.τ, inst.ffnet)
  vs = [zeros(Ckdim^2) for (_, _, Ckdim) in cinfos]
  zs = [zeros(Ckdim^2) for (_, _, Ckdim) in cinfos]
  λs = [zeros(Ckdim^2) for (_, _, Ckdim) in cinfos]

  # Conclude and return
  init_time = time() - init_start_time
  params = AdmmParams(γ=γ, vs=vs, zs=zs, λs=λs, cinfos=cinfos)
  return params, init_time
end


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
  DJJt = D + (opts.lagρ * J * J')
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

# Nonnegative projection
function projRplus(γ :: VecF64)
  return max.(γ, 0)
end

# Project a vector onto the NSD cone
function projectNsd(vk :: VecF64)
  vdim = Int(round(sqrt(length(vk)))) # :)
  @assert length(vk) == vdim * vdim
  tmp = Symmetric(reshape(vk, (vdim, vdim)))
  eig = eigen(tmp)
  tmp = Symmetric(eig.vectors * Diagonal(min.(eig.values, 0)) * eig.vectors')
  return tmp[:]
end

# Use a solver to do the stepping X
function stepXsolver(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  model = Model(Mosek.Optimizer)
  set_optimizer_attribute(model, "QUIET", true)
  set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", opts.max_solve_time)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", opts.solver_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", opts.solver_tol)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", opts.solver_tol)
end


