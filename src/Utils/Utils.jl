# Some useful utilities, such as file IO
# Should only be used within main.jl or a similar file
module Utils

using LinearAlgebra
using Random
using Plots
pyplot()

using ..Stuff
include("nnet_parser.jl"); using .NNetParser
include("network_utils.jl");
include("plotting.jl");

#
export randomNetwork, runNetwork
export scaleMats, loadNeuralNetwork
export randomTrajectories
export plotRandomTrajectories, plotLines

end # End module

