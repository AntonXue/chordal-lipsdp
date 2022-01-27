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
  ζs :: Vector{Vector{Float64}}
  τs :: Vector{Vector{Float64}}
  μs :: Vector{Vector{Float64}}
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

function initParams(inst :: QueryInstance, opts :: AdmmOptions; randomized :: Bool = false)
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

  γ = randomized ? randn(sum(γdims)) : zeros(sum(γdims))

  ζs = [Hc(k, β+1, γdims) * γ for k in 1:p]
  
  vs = Vector{Any}()
  for k in 1:p
    Zkdim = sum(edims[k:k+β+1])
    ζk = randomized ? randn(Zkdim^2) : zeros(Zkdim^2)
    push!(vs, ζk)
  end

  τs = [randomized ? randn(length(vs[k])) : zeros(length(vs[k])) for k in 1:p]
  μs = [randomized ? randn(length(ζs[k])) : zeros(length(ζs[k])) for k in 1:p]

  α = opts.α
  params = AdmmParams(γ=γ, vs=vs, ζs=ζs, τs=τs, μs=μs, α=α, γdims=γdims, β=β)
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

  Ωinv = makeΩinv(β+1, edims)

  # Populate the Js and its dependencies
  for k in 1:p
    Jk_start_time = time()

    # Gather the tiling information
    kdims, tups = makeTilingInfo(k, β+1, edims)

    @assert length(tups) == length(ζinds[k])

    # Setup the overlap matrix
    Eck = Ec(k, β+1, edims)
    Ωkinv = Eck * Ωinv * Eck'

    # Aggregate each Yss[k] into Jk after slicing and inserting
    # Recalling that ζk[i] generates Y[k+j]
    Jk = Vector{Any}()
    zkaffparts = Vector{Any}()
    for (i, (j, (slicelow, slicehigh), (insertlow, inserthigh), jdims)) in enumerate(tups)
      Eslice = vcat([E(l, jdims) for l in slicelow:slicehigh]...)
      Eins = vcat([E(l, kdims) for l in insertlow:inserthigh]...)
      for Yil in Yss[k+j]
        slicedYil = Eslice * Yil * Eslice'
        insYil = Eins' * slicedYil * Eins
        insYil = insYil .* Ωkinv
        push!(Jk, vec(insYil))
      end

      # Compute the affine component also because it's convenient
      Yilaff = Yaffs[k+j]
      slicedYilaff = Eslice * Yilaff * Eslice'
      insYilaff = Eins' * slicedYilaff * Eins
      insYilaff = insYilaff .* Ωkinv
      push!(zkaffparts, vec(insYilaff))
    end

    # cache the Jacobian and affine components
    Jk = hcat(Jk...)

    # Jk = 10 * randn(size(Jk)) # FIXME FIXME FIXME
    push!(Js, Jk)

    zkaff = sum(zkaffparts)
    # zkaff = 10 * randn(length(zkaffparts[1])) # FIXME FIXME FIXME
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

# The primal residual value
function primalResidual(params :: AdmmParams, cache :: AdmmCache)
  vzs = Vector{Any}()
  ζγs = Vector{Any}()
  norm2 = 0
  for k in 1:params.p
    vzk = params.vs[k] - makezk(k, params.ζs[k], cache)
    push!(vzs, vzk)

    ζγk = params.ζs[k] - Hc(k, params.β+1, params.γdims) * params.γ
    push!(ζγs, ζγk)

    norm2 = norm2 + norm(vzk)^2 + norm(ζγk)^2
  end
  return vzs, ζγs, norm2
end

# Calculate the vectorized Zk
function makezk(k :: Int, ζk, cache :: AdmmCache)
  return cache.Js[k] * ζk + cache.zaffs[k]
end

# Project onto the nonnegative orthant
function projectΓ(γ :: Vector{Float64})
  return max.(γ, 0)
  # return [γ[1]; max.(γ[2:end], 0)]
end

#
function stepγ(params :: AdmmParams, cache :: AdmmCache)
  tmp = [Hc(k, params.β+1, params.γdims)' * (params.ζs[k] + (params.μs[k] / params.α)) for k in 1:params.p]
  tmp = sum(tmp)
  tmp = tmp - (e(1, sum(params.γdims)) / params.α)
  tmp = cache.Dinv .* tmp
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

# The X = {γ, v1, ..., vp} variable updates
function stepX(params :: AdmmParams, cache :: AdmmCache)
  println("stepX!")
  new_γ = stepγ(params, cache)
  new_vs = Vector([stepvk(k, params, cache) for k in 1:params.p])
  return (new_γ, new_vs)
end

# The solver version
function stepXsolver(params :: AdmmParams, cache :: AdmmCache)
  println("stepXsolver!")
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-9
  ))

  lenγ = length(params.γ)
  ζs = params.ζs
  τs = params.τs
  μs = params.μs
  α = params.α

  β = params.β
  p = params.p
  γdims = params.γdims

  var_γ = @variable(model, [1:lenγ])
  @constraint(model, var_γ[1:lenγ] .>= 0)
  
  var_Vs = Vector{Any}()
  for k in 1:params.p
    vkdim = Int(round(sqrt(length(params.vs[k]))))
    var_Vk = @variable(model, [1:vkdim, 1:vkdim], Symmetric)
    @SDconstraint(model, var_Vk <= 0)
    push!(var_Vs, var_Vk)
  end

  # The relevant parts of the augmented Lagrangian

  γparts = [ζs[k] - Hc(k, β+1, γdims) * var_γ + (μs[k] / α) for k in 1:p]
  Vparts = [vec(var_Vs[k]) - makezk(k, ζs[k], cache) + (τs[k] / α) for k in 1:p]

  γnorm2s = Vector{Any}()
  Vnorm2s = Vector{Any}()
  for k in 1:p
    γknorm = @variable(model)
    Vknorm = @variable(model)

    @constraint(model, [γknorm; γparts[k]] in SecondOrderCone())
    @constraint(model, [Vknorm; Vparts[k]] in SecondOrderCone())

    push!(γnorm2s, γknorm^2)
    push!(Vnorm2s, Vknorm^2)
  end


  # For satisfiability
  # L = (α / 2) * (sum(Vnorm2s) + sum(γnorm2s))

  # For optimality
  L = var_γ[1] + (α / 2) * (sum(Vnorm2s) + sum(γnorm2s))

  @objective(model, Min, L)
  optimize!(model)

  new_γ = value.(var_γ)
  new_vs = [vec(value.(var_Vs[k])) for k in 1:p]
  return (new_γ, new_vs)
end


#
function stepY(params :: AdmmParams, cache :: AdmmCache)
  new_ζs = Vector([stepζk(k, params, cache) for k in 1:params.p])
  return new_ζs
end

#
function stepZ(params :: AdmmParams, cache :: AdmmCache)
  new_τs = Vector([stepτk(k, params, cache) for k in 1:params.p])
  new_μs = Vector([stepμk(k, params, cache) for k in 1:params.p])
  return (new_τs, new_μs)
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
  param_hist = Vector{Any}()

  # push!(param_hist, deepcopy(iter_params))

  iters_run = 0
  total_time = 0
  for t = 1:opts.max_iters

    step_start_time = time()

    xstart_time = time()
    # new_γ, new_vs = stepX(iter_params, cache)
    new_γ, new_vs = stepXsolver(iter_params, cache)
    iter_params.γ = new_γ
    iter_params.vs = new_vs
    xtotal_time = time() - xstart_time

    ystart_time = time()
    new_ζs = stepY(iter_params, cache)
    iter_params.ζs = new_ζs
    ytotal_time = time() - ystart_time

    zstart_time = time()
    new_τs, new_μs = stepZ(iter_params, cache)
    iter_params.τs = new_τs
    iter_params.μs = new_μs
    ztotal_time = time() - zstart_time

    # Coalesce all the time statistics
    step_total_time = time() - step_start_time
    total_time = total_time + step_total_time
    all_times = round.((xtotal_time, ytotal_time, ztotal_time, step_total_time, total_time), digits=2)
    if opts.verbose; println("step[" * string(t) * "/" * string(opts.max_iters) * "]" * " time: " * string(all_times)) end

    println("\tρ: " * string(iter_params.γ[1]))

    iters_run = t
    push!(param_hist, deepcopy(iter_params))


    vzs, ζγs, resnorm2 = primalResidual(iter_params, cache)
    println("\tresnorm2: " * string(resnorm2))


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

    println("")
    γdims = iter_params.γdims
    for k in 1:iter_params.p
      γk = round.(E(k, γdims) * iter_params.γ, digits=3)
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

    println("\tρ: " * string(iter_params.γ[1]))

    println("\n *** \t\t\t *** \t\t\t *** \n")

    =#

    # End dump space
    

  end

  return param_hist, iter_params, isγSat(iter_params, cache, opts), iters_run, total_time
end

# Call this
function run(inst :: QueryInstance, opts :: AdmmOptions)
  start_time = time()
  start_params = initParams(inst, opts)

  precompute_start_time = time()
  cache = precompute(start_params, inst, opts)
  precompute_time = time() - precompute_start_time

  param_hist, new_params, issat, iters_run, admm_iters_time = admm(start_params, cache, opts)

  # ρ = new_params.ρ
  ρ = new_params.γ[1]

  total_time = time() - start_time
  output = SolutionOutput(
            model = new_params,
            summary = "iters run: " * string(iters_run),
            status = issat ? "OPTIMAL" : "UNKNOWN",
            objective_value = "ρ: " * string(ρ),
            total_time = total_time,
            setup_time = precompute_time,
            solve_time = admm_iters_time)
  return (param_hist, output)
end


export AdmmOptions
export makezk, projectΓ, projectNsd, stepX, stepY, stepZ
export initParams, precompute

end # End module

