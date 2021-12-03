# Implement ADMM
module AdmmLipSdp

using ..Header
using ..Common
using Parameters
using LinearAlgebra
using JuMP
using Mosek
using MosekTools

#
@with_kw struct AdmmOptions
  max_iters :: Int = 20
  beign_check_at_iter :: Int = 5
  check_every_k_iters :: Int = 2
  nsd_tol :: Float64 = 1e-4
  α :: Float64 = 1.0
  verbose :: Bool = false
end

#
@with_kw mutable struct AdmmParams
  γ :: Vector{Float64}
  vs :: Vector{Vector{Float64}}
  ζs :: Vector{Vector{Float64}}
  τs :: Vector{Vector{Float64}}
  μs :: Vector{Vector{Float64}}
  α :: Float64
  γdims :: Vector{Int}
  p :: Int = length(γdims)
end

#
@with_kw mutable struct AdmmCache
  ζinds :: Vector{Vector{Int}} # The γ indices used by each ζk
  Js :: Vector{Matrix{Float64}}
  zaffs :: Vector{Vector{Float64}}
  Jtzaffs :: Vector{Vector{Float64}}
  I_JtJ_invs :: Vector{Matrix{Float64}}
  Dinv :: Vector{Float64}
end

# Initialize zero parameters of the appropriat esize

function initParams(inst :: QueryInstance, opts :: AdmmOptions)
  @assert inst.net isa FeedForwardNetwork
  ffnet = inst.net
  edims = ffnet.edims
  fdims = ffnet.fdims
  p = inst.p
  β = inst.β

  # The γdimension
  γdims = Vector{Int}(zeros(p))
  for k in 1:p
    γkdim = sum(fdims[k:k+β]) ^2
    if k == 1; γkdim += 1 end
    γdims[k] = γkdim
  end

  γ = zeros(sum(γdims))
  ζs = [Hc(k, β, γdims) * γ for k in 1:p]
  
  vs = Vector{Any}()
  for k in 1:p
    Zkdim = sum(edims[k:k+β])
    push!(vs, zeros(Zkdim^2))
  end

  τs = [zeros(length(vs[k])) for k in 1:p]
  μs = [zeros(length(ζs[k])) for k in 1:p]
  α = opts.α
  params = AdmmParams(γ=γ, vs=vs, ζs=ζs, τs=τs, μs=μs, α=α, γdims=γdims)
  return params
end

#
function precompute(params :: AdmmParams, inst :: QueryInstance, opts :: AdmmOptions)
  @assert inst.net.nettype isa ReluNetwork || inst.net.nettype isa TanhNetwork
  @assert params.p == inst.p

  if opts.verbose; println("precompute running") end

  γdims = params.γdims
  β = inst.β
  ffnet = inst.net
  p = params.p
  edims = ffnet.edims
  L = ffnet.L

  # The γ indices touched by each ζk
  ζinds = [Hcinds(k, β, γdims) for k in 1:p]

  # Yss[k] the non-affine components of Yk, Yaffs[k] is the affine components of Yk
  Yss = Vector{Any}()
  Yaffs = Vector{Any}()
  for k in 1:p
    yk_start_time = time()

    # Need to construct the affine component first
    Ykaff = makeYk(k, β, zeros(γdims[k]), ffnet, inst.pattern)
    push!(Yaffs, Ykaff)

    # Now construct the other stuff
    Ykparts = Vector{Any}()
    for j in 1:γdims[k]
      tmp = makeYk(k, β, e(j, γdims[k]), ffnet, inst.pattern)
      Ykj = tmp - Ykaff
      push!(Ykparts, Ykj)
    end

    yk_time = round.(time() - yk_start_time, digits=2)
    if opts.verbose; println("precompute: Yss[" * string(k) * "/" * string(p) * "], time: " * string(yk_time)) end
  end

  Js = Vector{Any}()
  zaffs = Vector{Any}()
  Jtzaffs = Vector{Any}()
  I_JtJ_invs = Vector{Any}()

  # Populate the Js and its dependencies
  for k in 1:p
    Jk_start_time = time()

    # Gather the tiling information
    kdims, tups = makeTilingInfo(k, β+1, edims)

    for t in tups; println(t) end

    println("numtups: " * string(length(tups)) * ", inds: " * string(length(ζinds[k])))

    @assert length(tups) == length(ζinds[k])

    # Aggregate each Yss[k] into Jk after slicing and inserting
    Jk = Vector{Any}()
    for (i, (j, (slicelow, slicehigh), (insertlow, inserthigh), jdims)) in enumerate(tups)
      Eslice = vcat([E(l, jdims) for l in slicelow:slicehigh]...)
      Eins = vcat([E(l, kdims) for l in insertlow:inserthigh]...)
      for Yil in Yss[i]
        slicedYil = Eslice * Yil * Eslice'
        vecYil = vec(Eins' * slicedYil * Eins)
        push!(Jk, vecYil)
      end
    end

    Jk = hcat(Jk...)


    Jk_time = round.(time() - Jk_start_time, digits=2)
    if opts.verbose; println("precompute: Js[" * string(k) * "/" * string(p) * "], time: " * string(Jk_time)) end
  end

end


# Project onto the nonnegative orthant
function projectΓ(γ :: Vector{Float64})
  return max.(γ, 0)
end

# The γ step
function stepγ(params :: AdmmParams, cache :: AdmmCache)
  # TODO: optimize
  tmp = [Hc(k, params.β, params.γdims)' * (params.ζs[k] + (params.μs[k] / params.α)) for k in 1:params.p]
  tmp = sum(tmp) .* cache.Dinv
  return projectΓ(tmp)
end

# Project a vector onto the negative semidefinite cone
function projectNsd(vk :: Vector{Float64})
  dim = Int(round(sqrt(length(vk)))) # :)
  @assert length(vk) == dim * dim
  tmp = Symmetric(reshape(vk, (dim, dim)))
  eig = eigen(tmp)
  tmp = Symmetric(eig.vectors * Diagonal(min.(eig.values, 0)) * eig.vectors')
  return tmp[:]
end

# The vk step
function stepvk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
  tmp = makezk(k, params.ζs[k], cache)
  tmp = tmp - (params.τs[k] / params.α)
  return projectNsd(tmp)
end

# The ζk step
function stepζk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
  tmp = cache.Js[k]' * params.vs[k]
  tmp = tmp + Hc(k, params.γdims) * params.γ
  tmp = tmp + (cache.Js[k]' * params.τs[k] - params.μs[k]) / params.ρ
  tmp = tmp - cache.Jtzaffs[k]
  tmp = cache.I_JtJ_invs[k] * tmp
  return tmp
end

# The τk update
function stepτk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
  tmp = params.vs[k] - zk(k, params.ζs[k], cache)
  tmp = params.τs[k] + params.ρ * tmp
  return tmp
end

# The μk update
function stepμk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
  tmp = params.ζs[k] - Hc(k, params.γdims) * params.γ
  tmp = params.μs[k] + params.ρ * tmp
  return tmp
end


export AdmmOptions
export initParams, precompute

end # End module

