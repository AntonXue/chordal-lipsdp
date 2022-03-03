# Some useful utilities, such as file IO
# Should only be used within main.jl or a similar file
module Utils

using LinearAlgebra
using Random
using Plots

using ..Stuff
include("nnet_parser.jl"); using .NNetParser

pyplot()

# Generate a random network given the desired dimensions at each layer
function randomNetwork(xdims :: VecInt; activ :: Activation = ReluActivation(), σ :: Float64 = 1.0)
  @assert length(xdims) > 1
  Ms = Vector{Any}()
  for k = 1:length(xdims) - 1
    # Width is xdims[k]+1 because Mk = [Wk bk]
    Mk = randn(xdims[k+1], xdims[k]+1) * σ
    push!(Ms, Mk)
  end
  return NeuralNetwork(activ=activ, xdims=xdims, Ms=Ms)
end

# Run a feedforward net on an initial input and give the output
function runNetwork(x1, ffnet :: NeuralNetwork)
  function ϕ(x)
    if ffnet.activ isa ReluActivation
      return max.(x, 0)
    elseif ffnet.activ isa TanhActivation
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
function randomTrajectories(N :: Int, ffnet :: NeuralNetwork)
  Random.seed!(1234)
  x1s = 2 * (rand(ffnet.xdims[1], N) .- 0.5) # Unit box
  # x1s = x1s ./ norm(x1s) # Unit vectors
  xfs = [runNetwork(x1s[:,k], ffnet) for k in 1:N]
  return xfs
end

# Plot some data to a file
function plotRandomTrajectories(N :: Int, ffnet :: NeuralNetwork, imgfile="~/Desktop/hello.png")
  # Make sure we can actually plot these in 2D
  @assert ffnet.xdims[end] == 2
  xfs = randomTrajectories(N, ffnet)
  d1s = [xf[1] for xf in xfs]
  d2s = [xf[2] for xf in xfs]
  p = scatter(d1s, d2s, markersize=2, alpha=0.3)
  savefig(p, imgfile)
end

# Plot different line data
# Get data of form (label1, ys1), (label2, ys2), ...
function plotLines(xs, labeled_lines :: Vector{Tuple{String, VecF64}};
                   title = "title", ylogscale = false, saveto = "~/Desktop/foo.png")
  # Make sure we have a consistent number of data
  @assert all(lys -> length(xs) == length(lys[2]), labeled_lines)
  plt = plot(title=title)
  colors = theme_palette(:auto)
  for (i, (lbl, ys)) in enumerate(labeled_lines)
    if ylogscale
      plot!(xs, ys, label=lbl, color=colors[i], yscale=:log10)
    else
      plot!(xs, ys, label=lbl, color=colors[i])
    end
  end
  savefig(plt, saveto)
  return plt
end

# Convert NNet to NeuralNetwork + BoxInput
function loadNeuralNetwork(nnet_filepath :: String)
  nnet = NNetParser.NNet(nnet_filepath)
  Ms = [[nnet.weights[k] nnet.biases[k]] for k in 1:nnet.numLayers]
  xdims = Vector{Int}(nnet.layerSizes)
  ffnet = NeuralNetwork(activ=ReluActivation(), xdims=xdims, Ms=Ms)
  return ffnet
end

function scaleNeuralNetwork(ffnet :: NeuralNetwork, scale :: Float64)
  @assert scale > 0
  scaledMs = ffnet.Ms .* scale
  new_ffnet = NeuralNetwork(activ=ffnet.activ, xdims=ffnet.xdims, Ms=scaledMs)
  return new_ffnet
end

#
export randomNetwork
export runNetwork, randomTrajectories
export plotRandomTrajectories, plotLines
export loadNeuralNetwork, scaleNeuralNetwork

end # End module

