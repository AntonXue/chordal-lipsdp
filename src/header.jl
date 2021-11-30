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
  mdims :: Vector{Int} = xdims[1:end-1]
  λdims :: Vector{Int} = mdims[2:end]

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
abstract type TkSparsity end

struct TkNoSparsity <: TkSparsity end
struct TkOnePerClique <: TkSparsity end
struct TkOnePerLayer <: TkSparsity end
struct TkOnePerNeuron <: TkSparsity end

@with_kw struct TkBanded <: TkSparsity
  α :: Int
  @assert α >= 0
end

# A solve instance
@with_kw struct QueryInstance
  net :: FeedForwardNetwork
  β :: Int = 0 # The outer sparsity pattern
  innerSparsity :: TkSparsity
  @assert β >= 0
end

# A solve output
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
export TkSparsity, TkNoSparsity, TkOnePerClique, TkOnePerLayer, TkOnePerNeuron, TkBanded
export QueryInstance, QueryOptions
export SolutionOutput

end # End module

