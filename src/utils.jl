# Some useful utilities, such as file IO
# Should only be used within main.jl or a similar file
module Utils

using ..Header
using LinearAlgebra
using DelimitedFiles
using Random
using Plots

# Write some data (hopefully Float64-based) to a file
function fileWriteFloat64(data, file :: String)
  open(file, "w") do io
    writedlm(io, data, ',')
  end
end

# Read some Float64-based data from a file
function fileReadFloat64(file :: String)
  data = readdlm(file, ',', Float64)
  return data
end

# Generate a random network given the desired dimensions at each layer
function randomNetwork(xdims :: VecInt; type :: NetworkType = ReluNetwork(), σ :: Float64 = 1.0)
  @assert length(xdims) > 1
  Ms = Vector{Any}()
  for k = 1:length(xdims) - 1
    # Width is xdims[k]+1 because Mk = [Wk bk]
    Mk = randn(xdims[k+1], xdims[k]+1) * σ
    push!(Ms, Mk)
  end
  return FeedForwardNetwork(type=type, xdims=xdims, Ms=Ms)
end

# Run a feedforward net on an initial input and give the output
function runNetwork(x1, ffnet :: FeedForwardNetwork)
  function ϕ(x)
    if ffnet.type isa ReluNetwork
      return max.(x, 0)
    elseif ffnet.type isa TanhNetwork
      return tanh.(x)
    else
      error("unsupported network: " * string(ffnet))
    end
  end

  xk = x1
  # Run through each layer
  for Mk in ffnet.Ms[1:end-1]
    xk = Mk * [xk; 1]
    xk = ϕ(xk)
  end
  # Then the final layer does not have an activation
  xk = ffnet.Ms[end] * [xk; 1]
  return xk
end

# Generate trajectories from a unit box
function randomTrajectories(N :: Int, ffnet :: FeedForwardNetwork)
  Random.seed!(1234)
  x1s = 2 * (rand(ffnet.xdims[1], N) .- 0.5) # Unit box
  # x1s = x1s ./ norm(x1s) # Unit vectors
  xfs = [runNetwork(x1s[:,k], ffnet) for k in 1:N]
  return xfs
end

# Plot some data to a file
function runAndPlotRandomTrajectories(N :: Int, ffnet :: FeedForwardNetwork, imgfile="~/Desktop/hello.png")
  # Make sure we can actually plot these in 2D
  @assert ffnet.xdims[end] == 2

  xfs = randomTrajectories(N, ffnet)
  d1s = [xf[1] for xf in xfs]
  d2s = [xf[2] for xf in xfs]
  
  p = scatter(d1s, d2s, markersize=2, alpha=0.3)
  savefig(p, imgfile)
end

#
export fileWriteFloat64, fileReadFloat64
export randomNetwork
export runNetwork, randomTrajectories, runAndPlotRandomTrajectories

end # End module

