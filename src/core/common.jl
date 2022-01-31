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

# Block matrix of [e(k); e(k+1); ...; e(k+size-1)]
# Assertion fails on out-of-bounds
function Ec(k :: Int, size :: Int, dim :: Int)
  @assert k >= 1 && size >= 1
  @assert 1 <= k + size - 1 <= dim
  ets = [e(j, dim)' for j in k:(k+size-1)]
  return vcat(ets...)
end

# Make the A block matrix
function makeA(ffnet :: FeedForwardNetwork)
  Ws = [M[1:end, 1:end-1] for M in ffnet.Ms]
  A = sum(E(k, ffnet.fdims)' * Ws[k] * E(k, ffnet.edims) for k in 1:(ffnet.K-1))
  return A
end

# Make the B block matrix
function makeB(ffnet :: FeedForwardNetwork)
  B = sum(E(k, ffnet.fdims)' * E(k+1, ffnet.edims) for k in 1:(ffnet.K-1))
  return B
end

# Calculate how long the γ should be given a particular τ
function γlength(Tdim :: Int, τ :: Int)
  @assert 0 <= τ
  return sum((Tdim-τ):Tdim)
end

# The T
function makeT(Tdim :: Int, γ, τ :: Int)
  @assert length(γ) == γlength(Tdim, τ)
  T = diagm(γ[1:Tdim])
  # T = Matrix(Diagonal(γ[1:Tdim]))
  # T = I(Tdim)
  if τ > 0
    ijs = [(i,j) for i in 1:(Tdim-1) for j in (i+1):Tdim if j-i <= τ]
    δts = [e(i,Tdim)' - e(j,Tdim)' for (i,j) in ijs]
    Δ = vcat(δts...)

    # Given a pair i,j calculate its relative index in the γ vector
    pair2ind(i,j) = sum((Tdim-(j-i)+1):Tdim) + i
    v = vec([γ[pair2ind(i,j)] for (i,j) in ijs])
    T += Δ' * (v .* Δ)
  end
  return T
end

# Construct M1, or smaller variants depending on what is queried with
function makeM1(T, A, B, ffnet :: FeedForwardNetwork)
  @assert ffnet.type isa ReluNetwork || ffnet.type isa TanhNetwork
  low, high = 0, 1
  _R11 = -2 * low * high * A' * T * A
  _R12 = (low + high) * A' * T * B
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

# Calculate the start and size of each clique
function makeCliqueInfos(b, ffnet :: FeedForwardNetwork)
  edims = ffnet.edims
  Zdim = sum(edims)
  clique_infos = Vector{Tuple{Int, Int, Int}}()
  for k in 1:length(edims)
    start = sum(edims[1:k-1])+1
    Ckdim = edims[k] + edims[k+1] + b
    if start + Ckdim - 1 >= Zdim
      push!(clique_infos, (k, start, Zdim-start+1))
      break
    else
      push!(clique_infos, (k, start, Ckdim))
    end
  end
  return clique_infos
end


# A function for making T; smaller instances of Tk can also be made using this
export e, E, Ec
export makeA, makeB
export γlength, makeT, makeM1, makeM2
export makeCliqueInfos

end # End module

