# Define a bunch of types that we would like to have around
module Header

using Parameters

# Different types of networks
abstract type NetworkType end
struct ReluNetwork <: NetworkType end
struct TanhNetwork <: NetworkType end

# Generic neural network supertype
abstract type NeuralNetwork end 
@with_kw struct FeedForwardNetwork <: NeuralNetwork
  # The type of the network
  nettype :: NetworkType

  # The state vector dimension at start of each layer
  xdims :: Vector{Int}

  # The dimensions for the T
  edims :: Vector{Int} = xdims[1:end-1]
  fdims :: Vector{Int} = edims[2:end]

  # Each M[K] == [Wk bk]
  Ms :: Vector{Matrix{Float64}}
  K :: Int = length(Ms)
  L :: Int = K - 1

  # Assert a non-trivial structural integrity of the network
  @assert length(xdims) >= 3
  @assert length(xdims) == K + 1
  @assert all([size(Ms[k]) == (xdims[k+1], xdims[k]+1) for k in 1:K])
end

# Different patterns of sparsity within each Tk
abstract type TPattern end
struct NoPattern <: TPattern end
struct OnePerNeuronPattern <: TPattern end
struct OnePerCliquePattern <: TPattern end
struct OnePerLayerPattern <: TPattern end
@with_kw struct BandedPattern <: TPattern
  band :: Int
  @assert band >= 0
end

# A query instance
@with_kw struct QueryInstance
  net :: FeedForwardNetwork
  β :: Int = 0 # The outer sparsity pattern must be banded
  p :: Int = net.L - β # The number of maximal cliques
  pattern :: TPattern # In addition, there might be internal sparsity patterns
  @assert 0 <= β <= net.L - 1
end

# A solution output
@with_kw struct SolutionOutput{M, S, V}
  model :: M
  summary :: S
  status :: String
  objective_value :: V
  total_time :: Float64
  setup_time :: Float64
  solve_time :: Float64
end

export NetworkType, ReluNetwork, TanhNetwork
export NeuralNetwork, FeedForwardNetwork
export TPattern, NoPattern, OnePerCliquePattern, OnePerLayerPattern, OnePerNeuronPattern, BandedPattern
export QueryInstance, QueryOptions
export SolutionOutput

end # End module

