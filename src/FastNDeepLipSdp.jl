module FastNDeepLipSdp

using LinearAlgebra
using Printf
using Random

include("Stuff/Stuff.jl");
include("Utils/Utils.jl");

import Reexport
Reexport.@reexport using .Stuff
Reexport.@reexport using .Utils

# The default Mosek options to use
DEFAULT_MOSEK_OPTS =
  Dict("QUIET" => true,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 1, # seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-9,
       "INTPNT_CO_TOL_PFEAS" => 1e-9,
       "INTPNT_CO_TOL_DFEAS" => 1e-9)

# Solve a problem instance depending on what kind of options we give it
function solveLipschitz(ffnet::NeuralNetwork, opts::MethodOptions)
  inst = QueryInstance(ffnet=ffnet)
  soln = runQuery(inst, opts) # Multiple dispatch based on opts type
  return soln
end

export DEFAULT_MOSEK_OPTS
export solveLipschitz

end

