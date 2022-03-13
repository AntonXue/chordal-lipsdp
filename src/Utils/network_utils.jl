
# Generate a random network given the desired dimensions at each layer
function randomNetwork(xdims::VecInt; activ::Activation = ReluActivation(), σ::Float64 = 1.0)
  @assert length(xdims) > 1
  Ms = Vector{Any}()
  for k = 1:length(xdims) - 1
    # Width is xdims[k]+1 because Mk = [Wk bk]
    Mk = randn(xdims[k+1], xdims[k]+1) * σ
    push!(Ms, Mk)
  end
  return NeuralNetwork(activ=activ, xdims=xdims, Ms=Ms)
end

#
function getϕ(activ::Activation)
  if activ isa ReluActivation
    return x -> max.(x, 0)
  elseif activ isa TanhActivation
    return x -> tanh.(x, 0)
  else
    error("unsupported activation $(activ)")
  end
end

# Run a feedforward net on an initial input and give the output
function runNetwork(x1, ffnet::NeuralNetwork)
  ϕ = getϕ(ffnet.activ)
  xk = x1
  # Run through each layer
  for Mk in ffnet.Ms[1:end-1]; xk = ϕ(Mk * [xk; 1]) end
  # Then the final layer does not have an activation
  xk = ffnet.Ms[end] * [xk; 1]
  return xk
end

# Scale all the weight matrices to match some desired opnorm
function scaleMats(Ms::Vector{MatF64}, target_opnorm::Float64)
  @assert target_opnorm > 0
  scales = target_opnorm ./ opnorm.(Ms)
  return (Ms .* scales), scales
end

# Convert NNet to NeuralNetwork
function loadNeuralNetwork(nnet_filepath::String)
  nnet = NNetParser.NNet(nnet_filepath)
  Ms = [[nnet.weights[k] nnet.biases[k]] for k in 1:nnet.numLayers]
  xdims = Vector{Int}(nnet.layerSizes)
  ffnet = NeuralNetwork(activ=ReluActivation(), xdims=xdims, Ms=Ms)
  return ffnet
end

# Scaled version of the load
function loadNeuralNetwork(nnet_filepath::String, target_opnorm::Float64)
  ffnet = loadNeuralNetwork(nnet_filepath)
  Ws, bs = [M[:,1:end-1] for M in ffnet.Ms], [M[:,end] for M in ffnet.Ms]
  αs = [target_opnorm / opnorm(W) for W in Ws]
  # αs = maximum(αs) * ones(ffnet.K)
  # αs = 2.0 * ones(ffnet.K)
  scaled_Ws = [αs[k] * Ws[k] for k in 1:ffnet.K]
  scaled_bs = [prod(αs[1:k]) * bs[k] for k in 1:ffnet.K]
  scaled_Ms = [[scaled_Ws[k] scaled_bs[k]] for k in 1:ffnet.K]
  scaled_ffnet = NeuralNetwork(activ=ffnet.activ, xdims=ffnet.xdims, Ms=scaled_Ms)
  return scaled_ffnet, αs
end

