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
function makeAc(k :: Int, β :: Int, ffnet :: FeedForwardNetwork)
  @assert k >= 1 && β >= 0
  @assert 1 <= k + β <= ffnet.K - 1
  mslice = ffnet.mdims[k:(k+β)]
  λslice = ffnet.λdims[k:(k+β)]
  Ack = sum(E(j, λslice)' * ffnet.Ms[k+j-1][1:end, 1:end-1] * E(j, mslice) for j in 1:β+1)
  return Ack
end

# Construct the Y
function makeY(k :: Int, β :: Int, Tk, ffnet :: FeedForwardNetwork)
  @assert ffnet.nettype isa ReluNetwork or ffnet.nettype isa TanhNetwork
  @assert k >= 1 && β >= 0
  @assert 1 <= k + β <= ffnet.L

  λdims = ffnet.λdims
  Λkdim = sum(λdims[k:k+β])
  @assert size(Tk) == (Λkdim, Λkdim)

  # Generate the individual in the overlapping block matrix
  a, b = 0, 1
  Ack = makeAc(k, β, ffnet)
  _R11 = -2 * a * b * Ack' * Tk * Ack
  _R12 = (a + b) * Ack' * Tk
  _R22 = -2 * Tk

  # Expand each _R11 and add them
  ydims = ffnet.mdims[k:k+β+1]
  G1 = Ec(1, β, ydims)
  G2 = Ec(2, β, ydims)
  _Yk11 = G1' * _R11 * G1
  _Yk12 = G1' * _R12 * G2
  _Yk21 = _Yk12'
  _Yk22 = G2' * _R22 * G2
  Yk = _Yk11 + _Yk12 + _Yk21 + _Yk22
  return Yk
end

# Construct the initial Y
function makeYinit(β :: Int, ρ, ffnet :: FeedForwardNetwork)
  ydims = ffnet.mdims[1:1+β+1]
  G1 = E(1, ydims)
  Yinit = -ρ * G1' * G1
  return Yinit
end

# Construct the final Y
function makeYfinal(β :: Int, ffnet :: FeedForwardNetwork)
  ydims = ffnet.mdims[(ffnet.K-β-1):ffnet.K]
  WK = ffnet.Ms[ffnet.K][1:end, 1:end-1]
  Gfinal = E(1+β+1, ydims)
  Yfinal = Gfinal' * WK' * WK * Gfinal
  return Yfinal
end

# A banded T matrix; if band >= dim then T is dense
function makeBandedT(dim :: Int, Λ; band :: Int = 0)
  @assert size(Λ) == (dim, dim)
  diagΛ = diagm(diag(Λ))
  if band > 0
    ijs = [(i, j) for i in 1:(dim-1) for j in (i+1):dim if abs(i-j) <= band]
    δts = [e(i, dim)' - e(j, dim)' for (i, j) in ijs]
    Δ = vcat(δts...)
    V = diagm(vec([Λ[i,j] for (i, j) in ijs]))
    T = diagΛ + Δ' * V * Δ
  else
    T = diagΛ
  end
  return T
end

# A function for making T; smaller instances of Tk can also be made using this
function makeT(Λdim :: Int, Λ, pattern :: TPattern)
  if pattern isa BandedPattern
    @assert size(Λ) == (Λdim, Λdim)
    return makeBandedT(Λdim, Λ, band=pattern.band)
  elseif pattern isa NoPattern
    @assert size(Λ) == (Λdim, Λdim)
    return makeBandedT(Λdim, Λ, band=Λdim)
  elseif pattern isa OnePerNeuronPattern
    @assert size(Λ) == (Λdim, Λdim)
    return makeBandedT(Λdim, Λ, band=0)
  else
    error("unsupported pattern: " * string(pattern))
  end
end

#
export e, E, Ec
export makeT, makeBandedT
export makeAc, makeY, makeYinit, makeYfinal

end # End module

