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

# Block matrix for [k; k+1; ...; k+b]
# Assertion fails on out-of-bounds
function Ec(k :: Int, b :: Int, dims :: Vector{Int})
  @assert k >= 1 && b >= 0
  @assert 1 <= k + b <= length(dims)
  Ejs = [E(j, dims) for j in k:(k+b)]
  return vcat(Ejs...)
end

# Consider the r-radius about k; i.e. [k-r; ...; k; ...; k+r]
# Allows radius to under/overflow
function Hc(k :: Int, r :: Int, dims :: Vector{Int})
  lendims = length(dims)
  @assert k >= 1 && r >= 0
  @assert 1 <= k <= lendims
  Ejs = [E(k+j, dims) for j in -r:r if 1 <= k+j <= lendims]
  return vcat(Ejs...)
end

# The indices relevant to Hc
function Hcinds(k :: Int, r :: Int, dims :: Vector{Int})
  lendims = length(dims)
  @assert k >= 1 && r >= 0
  @assert 1 <= k <= lendims
  return [k+j for j in -r:r if 1 <= k+j <= lendims]
end

# Construct blockdiag(W[k], ..., W[k+β]) by summing Fk' * Wk * Ek
function makeAck(k :: Int, β :: Int, ffnet :: FeedForwardNetwork)
  @assert k >= 1 && β >= 0
  @assert 1 <= k + β <= ffnet.L
  eds = ffnet.edims[k:(k+β)]
  fds = ffnet.fdims[k:(k+β)]
  Ack = sum(E(j, fds)' * ffnet.Ms[k+j-1][1:end, 1:end-1] * E(j, eds) for j in 1:β+1)
  return Ack
end

# Construct the X
function makeXk(k :: Int, β :: Int, Tk, ffnet :: FeedForwardNetwork)
  @assert ffnet.nettype isa ReluNetwork or ffnet.nettype isa TanhNetwork
  @assert k >= 1 && β >= 0
  @assert 1 <= k + β <= ffnet.L

  fdims = ffnet.fdims
  Λkdim = sum(fdims[k:k+β])
  @assert size(Tk) == (Λkdim, Λkdim)

  # Generate the individual in the overlapping block matrix
  seclow, sechigh = 0, 1
  Ack = makeAck(k, β, ffnet)
  _R11 = -2 * seclow * sechigh * Ack' * Tk * Ack
  _R12 = (seclow + sechigh) * Ack' * Tk
  _R22 = -2 * Tk

  # Expand each _R11 and add them
  ydims = ffnet.edims[k:k+β+1]
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
  ydims = ffnet.edims[1:1+β+1]
  G1 = E(1, ydims)
  Xinit = -ρ * G1' * G1
  return Xinit
end

# Construct the final X
function makeXfinal(β :: Int, ffnet :: FeedForwardNetwork)
  ydims = ffnet.edims[(ffnet.K-β-1):ffnet.K]
  WK = ffnet.Ms[ffnet.K][1:end, 1:end-1]
  Gend = E(1+β+1, ydims)
  Xfinal = Gend' * WK' * WK * Gend
  return Xfinal
end

# A banded T matrix with band b; if b >= dim then T is dense
function makeBandedT(dim :: Int, Λ; b :: Int = 0)
  @assert size(Λ) == (dim, dim)
  diagΛ = diagm(diag(Λ))
  if b > 0
    ijs = [(i, j) for i in 1:(dim-1) for j in (i+1):dim if abs(i-j) <= b]
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
    return makeBandedT(Λdim, Λ, b=pattern.band)
  elseif pattern isa NoPattern
    @assert size(Λ) == (Λdim, Λdim)
    return makeBandedT(Λdim, Λ, b=Λdim)
  elseif pattern isa OnePerNeuronPattern
    @assert size(Λ) == (Λdim, Λdim)
    return makeBandedT(Λdim, Λ, b=0)
  else
    error("unsupported pattern: " * string(pattern))
  end
end

# Make the Ys
function makeYk(k :: Int, β :: Int, γk, ffnet :: FeedForwardNetwork, pattern :: TPattern)
  @assert 1 <= k <= ffnet.L
  # Make the Xk first
  if k == 1
    γ1 = γk[2:end]
    Λdim = Int(round(sqrt(length(γ1))))
    @assert length(γ1) == Λdim^2
    Λ = reshape(γ1, (Λdim, Λdim))
  else
    Λdim = Int(round(sqrt(length(γk))))
    @assert length(γk) == Λdim^2
    Λ = reshape(γk, (Λdim, Λdim))
  end
  Tk = makeT(Λdim, Λ, pattern)
  Xk = makeXk(k, β, Tk, ffnet)

  # Then check to see if we should add the Xinit
  if k == 1; Xk += makeXinit(β, γk[1], ffnet) end

  # ... and the Xfinal in a separate condition in case p == 1
  if k + β == ffnet.L; Xk += makeXfinal(β, ffnet) end

  return Xk
end

# Overlap counter for a general banded structure
function makeΩ(b :: Int, dims :: Vector{Int})
  p = length(dims) - b
  @assert p >= 1
  Ω = zeros(sum(dims), sum(dims))
  for k in 1:p
    Eck = Ec(k, b, dims)
    height = size(Eck)[1]
    Ω += Eck' * (fill(1, (height, height))) * Eck
  end
  return Ω
end

# Make the ℧ matrix that is the "inverse" of Ω
function makeΩinv(b :: Int, dims :: Vector{Int})
  Ω = makeΩ(b, dims)
  Ωinv = 1 ./ Ω
  Ωinv[isinf.(Ωinv)] .= 0
  return Ωinv
end

# Calculate the relevant partition tuples
function makeTilingInfo(k :: Int, b :: Int, dims :: Vector{Int})
  lendims = length(dims)
  @assert k >= 1 && b >= 0
  @assert 1 <= k + b <= lendims
  jtups = Vector{Any}()
  for j in -b:b
    if (1 <= k + j <= lendims) && (1 <= k + j + b <= lendims)
      # The dimensions relevant to the j block
      jdims = dims[(k+j) : (k+j+b)]

      # The low/high slices of jdims
      slicelow = (j > 0) ? 1 : (1 - j)
      slicehigh = (j > 0) ? (b + 1 - j) : (b + 1)
      jslice = (slicelow, slicehigh)

      # The insertion places within the kdim block
      insertlow = (j > 0) ? (1 + j) : 1
      inserthigh = (j > 0) ? (b + 1) : (b + 1 + j)
      jinsert = (insertlow, inserthigh)

      push!(jtups, (j, jslice, jinsert, jdims))
    end
  end

  kdims = dims[k:k+b]
  return(kdims, jtups)
end

# Construct M1, or smaller variants depending on what is queried with
function makeM1(T, A, B, ffnet :: FeedForwardNetwork)
  @assert ffnet.nettype isa ReluNetwork || ffnet.nettype isa TanhNetwork
  seclow, sechigh = 0, 1
  _R11 = -2 * seclow * sechigh * A' * T * A
  _R12 = (seclow + sechigh) * A' * T * B
  _R21 = _R12'
  _R22 = -2 * B' * T * B
  M1 = _R11 + _R12 + _R21 + _R22
  return M1
end

# Construct M2
function makeM2(ρ, ffnet :: FeedForwardNetwork)
  E1 = E(1, ffnet.edims)
  EK = E(ffnet.K, ffnet.edims)
  WK = ffnet.Ms[ffnet.K][1:end, 1:end-1]
  _R1 = -ρ * E1' * E1
  _R2 = EK' * (WK' * WK) * EK
  M2 = _R1 + _R2
  return M2
end

#
export e, E, Ec, Hc, Hcinds
export makeT, makeBandedT
export makeAck, makeXk, makeXinit, makeXfinal
export makeYk
export makeΩ, makeΩinv
export makeM1, makeM2
export makeTilingInfo

end # End module

