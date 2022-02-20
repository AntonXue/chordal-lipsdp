using Parameters
using LinearAlgebra
using SparseArrays
using Printf

@with_kw struct AdmmSdpOptions
  max_iters :: Int = 200
  
  cholesky_reg_ε :: Float64 = 1e-2

  τ :: Int = 0
  lagρ :: Float64 = 1.0
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
  J :: SparseMatrixCSC{Float64, Int}
  zaff :: SparseVector{Float64, Int}
  Hs
  chL
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

  # Prepare to return
  cache_time = time() - cache_start_time
  cache = AdmmCache(J=J, zaff=zaff, Hs=0, chL=0)
  return cache, cache_time
end


