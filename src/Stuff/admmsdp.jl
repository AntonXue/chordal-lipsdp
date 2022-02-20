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


function precomputeAdmmCache(inst :: QueryInstance, params :: AdmmParams, opts :: AdmmSdpOptions)
  cache_start_time = time()

  # Some useful constants
  ffnet = inst.ffnet
  Zdim = sum(ffnet.edims)
  γdim = params.γdim

  # Compute the affine component first
  zaff = sparse(vec(makeZ(spzeros(γdim), opts.τ, ffnet)))

  J = spzeros(Zdim^2, γdim)
  for i in 1:γdim
    zk = sparse(vec(makeZ(e(i, γdim), opts.τ, ffnet))) - zaff
    J[:,i] = zk
  end

  # Prepare to return
  cache_time = time() - cache_start_time
  cache = AdmmCache(J=J, zaff=zaff, Hs=0, chL=0)
  return cache, cache_time
end


