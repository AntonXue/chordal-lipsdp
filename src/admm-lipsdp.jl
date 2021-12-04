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
  begin_check_at_iter :: Int = 5
  check_every_k_iters :: Int = 2
  nsd_tol :: Float64 = 1e-4
  α :: Float64 = 1.0
  verbose :: Bool = false
end

#
@with_kw mutable struct AdmmParams
  γ :: Vector{Float64}
  vs :: Vector{Vector{Float64}}
  ρ :: Float64
  ζs :: Vector{Vector{Float64}}
  τs :: Vector{Vector{Float64}}
  μs :: Vector{Vector{Float64}}
  η :: Float64
  α :: Float64
  γdims :: Vector{Int}
  β :: Int
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
  ρ = 1.0
  ζs = [Hc(k, β+1, γdims) * γ for k in 1:p]
  
  vs = Vector{Any}()
  for k in 1:p
    Zkdim = sum(edims[k:k+β+1])
    push!(vs, zeros(Zkdim^2))
  end

  τs = [zeros(length(vs[k])) for k in 1:p]
  μs = [zeros(length(ζs[k])) for k in 1:p]
  η = 1.0 # not zero, so that ρ doesn't instantly mess up
  α = opts.α
  params = AdmmParams(γ=γ, vs=vs, ρ=ρ, ζs=ζs, τs=τs, μs=μs, η=η, α=α, γdims=γdims, β=β)
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
  # Since this is tiled wrt Y, we need β+1
  ζinds = [Hcinds(k, β+1, γdims) for k in 1:p]

  # Yss[k] the non-affine components of Yk, Yaffs[k] is the affine components of Yk
  Yss = Vector{Any}()
  Yaffs = Vector{Any}()
  for k in 1:p
    yk_start_time = time()

    # Need to construct the affine component first
    Ykaff = makeYk(k, β, zeros(γdims[k]), ffnet, inst.pattern)

    # Now construct the other stuff
    Ykparts = Vector{Any}()
    for j in 1:γdims[k]
      tmp = makeYk(k, β, e(j, γdims[k]), ffnet, inst.pattern)
      Ykj = tmp - Ykaff
      push!(Ykparts, Ykj)
    end

    # Add the relevant parts
    push!(Yss, Ykparts)
    push!(Yaffs, Ykaff)

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

    @assert length(tups) == length(ζinds[k])

    # Aggregate each Yss[k] into Jk after slicing and inserting
    # Recalling that ζk[i] generates Y[k+j]
    Jk = Vector{Any}()
    zkaffparts = Vector{Any}()
    for (i, (j, (slicelow, slicehigh), (insertlow, inserthigh), jdims)) in enumerate(tups)
      Eslice = vcat([E(l, jdims) for l in slicelow:slicehigh]...)
      Eins = vcat([E(l, kdims) for l in insertlow:inserthigh]...)
      for Yil in Yss[k+j]
        slicedYil = Eslice * Yil * Eslice'
        vecYil = vec(Eins' * slicedYil * Eins)
        push!(Jk, vecYil)
      end

      # Compute the affine component also because it's convenient
      Yilaff = Yaffs[k+j]
      slicedYilaff = Eslice * Yilaff * Eslice'
      vecYilaff = vec(Eins' * slicedYilaff * Eins)
      push!(zkaffparts, vecYilaff)
    end

    # cache the Jacobian and affine components
    Jk = hcat(Jk...)
    push!(Js, Jk)

    zkaff = sum(zkaffparts)
    push!(zaffs, zkaff)

    # Jk' * zkaff
    _Jktzaff = Jk' * zkaff
    push!(Jtzaffs, _Jktzaff)

    # inv(I + Jk' * Jk)
    _I_JktJk_inv = inv(Symmetric(I + Jk' * Jk))
    push!(I_JtJ_invs, _I_JktJk_inv)

    Jk_time = round.(time() - Jk_start_time, digits=2)
    if opts.verbose; println("precompute: Js[" * string(k) * "/" * string(p) * "], time: " * string(Jk_time)) end
  end


  # Compute Dinv
  D = sum(Hc(k, β+1, γdims)' * Hc(k, β+1, γdims) for k in 1:p)
  D = diag(D)
  @assert minimum(D) >= 0
  Dinv = 1 ./ D

  # Complete the cache
  cache = AdmmCache(ζinds=ζinds, Js=Js, zaffs=zaffs, Jtzaffs=Jtzaffs, I_JtJ_invs=I_JtJ_invs, Dinv=Dinv)
  return cache
end

# Calculate the vectorized Zk
function makezk(k :: Int, ζk :: Vector{Float64}, cache :: AdmmCache)
  return cache.Js[k] * ζk + cache.zaffs[k]
end

# Project onto the nonnegative orthant
function projectΓ(γ :: Vector{Float64})
  return max.(γ, 0)
end

# The γ step
function stepγ(params :: AdmmParams, cache :: AdmmCache)
  # TODO: optimize
  tmp = [Hc(k, params.β+1, params.γdims)' * (params.ζs[k] + (params.μs[k] / params.α)) for k in 1:params.p]
  tmp = cache.Dinv .* sum(tmp)
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

# The ρ step
function stepρ(params :: AdmmParams, cache :: AdmmCache)
  # tmp = params.ρ - params.γ[1] + (params.η / params.α)
  # tmp = -1 / tmp
  tmp = (params.α * params.γ[1]) - params.η - 1
  tmp = tmp / params.α
  return tmp
end

# The ζk step
function stepζk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
  tmp = cache.Js[k]' * params.vs[k]
  tmp = tmp + Hc(k, params.β+1, params.γdims) * params.γ
  tmp = tmp + (cache.Js[k]' * params.τs[k] - params.μs[k]) / params.α
  tmp = tmp - cache.Jtzaffs[k]
  tmp = cache.I_JtJ_invs[k] * tmp
  return tmp
end

# The τk update
function stepτk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
  tmp = params.vs[k] - makezk(k, params.ζs[k], cache)
  tmp = params.τs[k] + params.α * tmp
  return tmp
end

# The μk update
function stepμk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
  tmp = params.ζs[k] - Hc(k, params.β+1, params.γdims) * params.γ
  tmp = params.μs[k] + params.α * tmp
  return tmp
end

# The η update
function stepη(params :: AdmmParams, cache :: AdmmCache)
  tmp = params.ρ - params.γ[1]
  tmp = params.η + params.α * tmp
  return tmp
end

# The X = {γ, v1, ..., vp} variable updates
function stepX(params :: AdmmParams, cache :: AdmmCache)
  new_γ = stepγ(params, cache)
  new_vs = Vector([stepvk(k, params, cache) for k in 1:params.p])
  return (new_γ, new_vs)
end

# The Y = {ρ, ζ1, ..., ζp} variable updates
function stepY(params :: AdmmParams, cache :: AdmmCache)
  new_ρ = stepρ(params, cache)
  new_ζs = Vector([stepζk(k, params, cache) for k in 1:params.p])
  return (new_ρ, new_ζs)
end

# The Z = {τ1, ..., τp, μ1, ..., μp} variable updates
function stepZ(params :: AdmmParams, cache :: AdmmCache)
  new_τs = Vector([stepτk(k, params, cache) for k in 1:params.p])
  new_μs = Vector([stepμk(k, params, cache) for k in 1:params.p])
  new_η = stepη(params, cache)
  return (new_τs, new_μs, new_η)
end

# Check that each Zk <= 0
function isγSat(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmOptions)
  #=
  if t > opts.begin_check_at_iter && mod(t, opts.check_every_k_iters) == 0
    if isγSat(params, cache, opts)
      if opts.verbose; println("Sat!") end
      return true
    end
  end
  return false
  =#
  return true
end

#
function admm(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmOptions)
  iter_params = deepcopy(params)
  iters_run = 0
  total_time = 0
  for t = 1:opts.max_iters
    step_start_time = time()

    xstart_time = time()
    new_γ, new_vs = stepX(iter_params, cache)
    iter_params.γ = new_γ
    iter_params.vs = new_vs
    xtotal_time = time() - xstart_time

    ystart_time = time()
    new_ρ, new_ζs = stepY(iter_params, cache)
    iter_params.ρ = new_ρ
    iter_params.ζs = new_ζs
    ytotal_time = time() - ystart_time

    zstart_time = time()
    new_τs, new_μs, new_η = stepZ(iter_params, cache)
    iter_params.τs = new_τs
    iter_params.μs = new_μs
    iter_params.η = new_η
    ztotal_time = time() - zstart_time

    # Coalesce all the time statistics
    step_total_time = time() - step_start_time
    total_time = total_time + step_total_time
    all_times = round.((xtotal_time, ytotal_time, ztotal_time, step_total_time, total_time), digits=2)
    if opts.verbose; println("step[" * string(t) * "/" * string(opts.max_iters) * "]" * " time: " * string(all_times)) end

    # println("\tρ: " * string(new_ρ))

    iters_run = t

    #=
    if shouldStop(t, iter_params, cache, opts)
      break
    end
    =#

    # Begin dump space
    #=
    println("γ:")
    display(round.(iter_params.γ, digits=3))
    println("\n")
    =#

    #=
  
    println("")
    γdims = iter_params.γdims
    for k in 1:iter_params.p
      γk = round.(E(1, γdims) * iter_params.γ, digits=3)
      println("γ[" * string(k) * "]: " * string(γk))
      println("")
    end

    println("")
    for k in 1:iter_params.p
      println("vs[" * string(k) * "]")
      vkdim = Int(round(sqrt(length(iter_params.vs[k]))))
      display(round.(reshape(iter_params.vs[k], (vkdim, vkdim)), digits=3))
      println("")
    end

    println("")
    for k in 1:iter_params.p
      ζk = round.(iter_params.ζs[k]', digits=3)
      println("ζs[" * string(k) * "]: " * string(ζk'))
      println("")
    end

    println("")
    for k in 1:iter_params.p
      τk = round.(iter_params.τs[k]', digits=3)
      println("τs[" * string(k) * "]: " * string(τk'))
      println("")
    end

    println("")
    for k in 1:iter_params.p
      μk = round.(iter_params.μs[k]', digits=3)
      println("μs[" * string(k) * "]: " * string(μk'))
      println("")
    end

    println("")
    println("η: " * string(iter_params.η))
    =#

    println("\tρ: " * string(iter_params.ρ))

    #=
    println("\n *** \t\t\t *** \t\t\t *** \n")
    =#


    # End dump space

  end

  return iter_params, isγSat(iter_params, cache, opts), iters_run, total_time
end

# Call this
function run(inst :: QueryInstance, opts :: AdmmOptions)
  start_time = time()
  start_params = initParams(inst, opts)

  precompute_start_time = time()
  cache = precompute(start_params, inst, opts)
  precompute_time = time() - precompute_start_time

  new_params, issat, iters_run, admm_iters_time = admm(start_params, cache, opts)

  ρ = new_params.ρ

  total_time = time() - start_time
  output = SolutionOutput(
            model = new_params,
            summary = "iters run: " * string(iters_run),
            status = issat ? "OPTIMAL" : "UNKNOWN",
            objective_value = "ρ: " * string(ρ),
            total_time = total_time,
            setup_time = precompute_time,
            solve_time = admm_iters_time)
  return output
end


export AdmmOptions
export initParams, precompute

end # End module

