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

# Construct the X
function makeX(k :: Int, β :: Int, Tk, ffnet :: FeedForwardNetwork)
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
  _Xk11 = G1' * _R11 * G1
  _Xk12 = G1' * _R12 * G2
  _Xk21 = _Xk12'
  _Xk22 = G2' * _R22 * G2
  Xk = _Xk11 + _Xk12 + _Xk21 + _Xk22
  return Xk
end

# Construct the initial X
function makeXinit(β :: Int, ρ, ffnet :: FeedForwardNetwork)
  ydims = ffnet.mdims[1:1+β+1]
  G1 = E(1, ydims)
  Xinit = -ρ * G1' * G1
  return Xinit
end

# Construct the final X
function makeXfinal(β :: Int, ffnet :: FeedForwardNetwork)
  ydims = ffnet.mdims[(ffnet.K-β-1):ffnet.K]
  WK = ffnet.Ms[ffnet.K][1:end, 1:end-1]
  Gfinal = E(1+β+1, ydims)
  Xfinal = Gfinal' * WK' * WK * Gfinal
  return Xfinal
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

# Overlap counter
function makeΩ(band :: Int, dims :: Vector{Int})
  p = length(dims) - band
  @assert p >= 1
  Ω = zeros(sum(dims), sum(dims))
  for k in 1:p
    Eck = Ec(k, band, dims)
    onesies = fill(1, size(Eck * Eck'))
    Ωk = Eck' * onesies * Eck
    Ω += Ωk
  end
  return Ω
end

#
function makeΩinv(band :: Int, dims :: Vector{Int})
  Ω = makeΩ(band, dims)
  Ωinv = 1 ./ Ω
  Ωinv[isinf.(Ωinv)] .= 0
  return Ωinv
end

# Selectors for γ
function Hc(k :: Int, band :: Int, j :: Int, γdims :: Vector{Int})
end

# Calculate the relevant partition tuples
function makePartitionTuples(k :: Int, band :: Int, dims :: Vector{Int})
  lendims = length(dims)
  @assert k >= 1 && band >= 0
  @assert 1 <= k + band <= lendims
  jtups = Vector{Any}()
  for j in -band:band
    if (1 <= k + j <= lendims) && (1 <= k + j + band <= lendims)
      # The dimensions relevant to the j block
      jdims = dims[(k+j) : (k+j+band)]

      # The low/high slices of jdims
      slicelow = (j > 0) ? 1 : 1-j
      slicehigh = (j > 0) ? band+1-j : band+1

      # The insertion places within the kdim block
      inslow = (j > 0) ? 1+j : 1
      inshigh = (j > 0) ? band+1 : band+1+j
      # println("at j: " * string(j) * ", slices: " * string((slicelow, slicehigh)) * ", ins: " * string((inslow, inshigh)) * ", jdims: " * string(jdims))
      push!(jtups, (j, slicelow, slicehigh, inslow, inshigh))
    end
  end

  kdims = dims[k:k+band]
  return(kdims, jtups)
end

# Construct M1, or smaller variants depending on what is queried with
function makeM1(T, A, B, ffnet :: FeedForwardNetwork)
  @assert ffnet.nettype isa ReluNetwork || ffnet.nettype isa TanhNetwork
  a, b = 0, 1
  _R11 = -2 * a * b * A' * T * A
  _R12 = (a + b) * A' * T * B
  _R21 = (a + b) * B' * T * A
  _R22 = -2 * B' * T * B
  M1 = _R11 + _R12 + _R21 + _R22
  return M1
end

# Construct M2
function makeM2(ρ, ffnet :: FeedForwardNetwork)
  E1 = E(1, ffnet.mdims)
  EK = E(ffnet.K, ffnet.mdims)
  WK = ffnet.Ms[ffnet.K][1:end, 1:end-1]
  _R1 = -ρ * E1' * E1
  _R2 = EK' * (WK' * WK) * EK
  M2 = _R1 + _R2
  return M2
end

#
export e, E, Ec
export makeT, makeBandedT
export makeAc, makeX, makeXinit, makeXfinal
export makeΩ, makeΩinv
export makeM1, makeM2
export makePartitionTuples

end # End module

