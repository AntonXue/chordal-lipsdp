# Define a bunch of types that we would like to have around
module Header

using Parameters
using Printf

# Helpful types
const VecInt = Vector{Int}
const VecF64 = Vector{Float64}
const MatF64 = Matrix{Float64}

# Different types of networks
abstract type NetworkType end
struct ReluNetwork <: NetworkType end
struct TanhNetwork <: NetworkType end

# Generic neural network supertype
abstract type NeuralNetwork end 
@with_kw struct FeedForwardNetwork <: NeuralNetwork
  # The type of the network
  type :: NetworkType

  # The state vector dimension at start of each layer
  xdims :: VecInt

  # The dimensions for the T
  edims :: VecInt = xdims[1:end-1]
  fdims :: VecInt = edims[2:end]

  # Each M[K] == [Wk bk]
  Ms :: Vector{MatF64}
  K :: Int = length(Ms)

  # Assert a non-trivial structural integrity of the network
  @assert length(xdims) >= 2
  @assert length(xdims) == K + 1
  @assert all([size(Ms[k]) == (xdims[k+1], xdims[k]+1) for k in 1:K])
end

# Query instance
@with_kw struct QueryInstance
  ffnet :: FeedForwardNetwork
end

# The solution that is to be output by an algorithm
@with_kw struct SolutionOutput{A, B, C, D}
  objective_value :: A
  values :: B
  summary :: C
  termination_status :: D
  total_time :: Float64
  setup_time :: Float64
  solve_time :: Float64
end

# Export suff
export VecInt, VecF64, MatF64
export NetworkType, ReluNetwork, TanhNetwork
export NeuralNetwork, FeedForwardNetwork
export QueryInstance
export SolutionOutput

end # End module

