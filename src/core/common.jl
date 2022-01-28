# Some functionalities that will be shared across different algorithms
module Common

using ..Header
using LinearAlgebra
using Printf

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

# Make the A block matrix
function makeA(ffnet :: FeedForwardNetwork)
  edims, fdims = ffnet.edims, ffnet.fdims
  Ws = [M[1:end, 1:end-1] for M in ffnet.Ms]
  A = sum(E(k, fdims)' * Ws[k] * E(k, edims) for k in 1:(ffnet.K-1))
  return A
end

# Make the B block matrix
function makeB(ffnet :: FeedForwardNetwork)
  edims, fdims = ffnet.edims, ffnet.fdims
  B = sum(E(k, fdims)' * E(k+1, edims) for k in 1:(ffnet.K-1))
  return B
end

# Calculate how long the λ should be given a particular tband
function λlength(dim :: Int, tband :: Int)
  @assert 0 <= tband
  return sum((dim-tband):dim)
end

# The T
function makeT(dim :: Int, λ, tband :: Int)
  @assert length(λ) == λlength(dim, tband)
  T = Diagonal(λ[1:dim])
  if tband > 0
    ijs = [(i,j) for i in 1:(dim-1) for j in (i+1):dim if j-i <= tband]
    δts = [e(i,dim)' - e(j,dim)' for (i,j) in ijs]
    Δ = vcat(δts...)

    # Given a pair i,j calculate its relative index in the λ vector
    pair2ind(i, j) = sum((dim-(j-i)+1):dim) + i
    v = vec([λ[pair2ind(i, j)] for (i,j) in ijs])
    T += Δ' * (v .* Δ)
  end
  return T
end

# Construct M1, or smaller variants depending on what is queried with
function makeM1(T, A, B, ffnet :: FeedForwardNetwork)
  @assert ffnet.type isa ReluNetwork || ffnet.type isa TanhNetwork
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


# A function for making T; smaller instances of Tk can also be made using this
export e, E
export makeA, makeB
export λlength, makeT, makeM1, makeM2

end # End module

