# Some functionalities that will be shared across different algorithms
module Common

using ..Header
using LinearAlgebra

# The ith basis vector
function e(i :: Int, dim :: Int)
  @assert 1 <= i <= dim
  e = zeros(dim)
  e[i] = 1
  return e
end

# The ith block index matrix
function E(i :: Int, dims :: Vector{Int})
  @assert 1 <= i <= length(dims)
  width = sum(dims)
  low = sum(dims[1:i-1]) + 1
  high = sum(dims[1:i])
  E = zeros(dims[i], width)
  E[1:dims[i], low:high] = I(dims[i])
  return E
end

# The kth β-banded block index matrix
function Ec(k :: Int, β :: Int, dims :: Vector{Int})
  @assert k >= 1 && β >= 0
  @assert 1 <= k + β <= length(dims)
  Ejs = [E(j, dims) for j in k:(k+β)]
  return vcat(Ejs...)
end

# Get the A(k, β) slice
function Ac(k :: Int, β :: Int, ffnet :: FeedForwardNetwork)
  @assert k >= 1 && β >= 0
  @assert 1 <= k + β <= ffnet.K - 1
  xslice = ffnet.mdims[k:(k+β)]
  λslice = ffnet.λdims[k:(k+β)]
  Aslice = sum(E(j, λslice)' * ffnet.Ms[k+j-1][1:end, 1:end-1] * E(j, xslice) for j in 1:β+1)
  return Aslice
end

# A banded Tk matrix with bandwidth α
function Tkbanded(dim :: Int, Λ; α :: Int = 0)
  @assert size(Λ) == (dim, dim)
  @assert 0 <= α <= dim - 1

  diagΛ = diagm(diag(Λ))
  if α > 0
    ijs = [(i, j) for i in 1:(dim-1) for j in (i+1):dim if abs(i-j) <= α]
    δts = [e(i, dim)' - e(j, dim)' for (i, j) in ijs]
    Δ = vcat(δts...)
    V = diagm(vec([Λ[i,j] for (i, j) in ijs]))
    T = diagΛ + Δ' * V * Δ
  else
    T = diagΛ
  end
  return T
end

#
export e, E, Ec
export Tkbanded
export Ac

end # End module


