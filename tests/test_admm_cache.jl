
using Random
using LinearAlgebra
using Printf

function testAdmmCache(; verbose = false)
  # Set up a random network
  Random.seed!(1234)
  xdims = [10; 10; 10; 10; 10; 10]
  ffnet = randomNetwork(xdims)
  inst = QueryInstance(ffnet=ffnet)
  τ = 2

  # Randomized γ
  γdim = γlength(τ, ffnet)
  γ = rand(γdim)

  # The Z under the typical construction
  Z = makeZ(γ, τ, ffnet)

  # Under the ADMM construction
  admm_opts = AdmmSdpOptions(τ=τ, verbose=verbose)
  admm_init_params, _ = initAdmmParams(inst, admm_opts)
  cache, _ = initAdmmCache(inst, admm_init_params, admm_opts)

  admmZ = reshape(cache.J * γ + cache.zaff, size(Z))

  maxdiff = maximum(abs.(Z - admmZ))
  @printf("maxdiff: %f\n", maxdiff)
  @assert maxdiff <= 1e-12

  return cache
end

