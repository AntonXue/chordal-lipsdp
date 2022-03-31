# chordal-lipsdp

This repository contains an implementation of Chordal-LipSDP, a chordally sparse formulation of the LipSDP technique for bounding Lipschitz constants of a feedforward neural network.


## Requirements and Installation
This codebase is tested with the following requirements:
- Julia 1.6.3
- MOSEK 9.3
- Python 3.8

It will be assumed that Julia is in your executable path. To install the necessary Julia packages, run
```
  julia scripts/install_pkgs.jl
```


## Generating Random Networks

Chordal-LipSDP currently reads networks in the [NNet](https://github.com/sisl/NNet) fileformat. To generate random networks you will need to clone [this repository](https://github.com/sisl/NNet) somewhere, e.g.
```
  mkdir -p ~/stuff/nv-repos
  cd ~/stuff/nv-repos
  git clone https://github.com/sisl/NNet.git
```
... and accordingly update the `NNET_PATH` value in `scripts/make_networks`. Then to generate random networks, run
```
 cd /path/to/chordal-lipsdp
 mkdir nnet
 python3 scripts/make_networks.py --nnetdir nnet
```
This will write a collection of random networks to the newly created `nnet/rand` directory.


## Running Evaluations

We will first create directory to dump information and then run the eval script
```
  mkdir dump
```
and then run Julia in interactive mode with
```
  julia -i scripts/run_evals.jl --nnetdir nnet --dumpdir dump
```
The evaluation script will first spend some time "warming up" --- by running the LipSDP, Chordal-LipSDP, Naive-Lip, and CP-Lip methods on a small network first --- in order to "precompile" everything. Observe that `scripts/run_evals.jl` has the paths of the randomized networks pre-defined. After the interpreter is loaded, evaluations can be queried in the REPL, for example:
```
  julia> runRandBatch(RAND_W10, :lipsdp)      # LipSDP on random netwoks of width 10
  julia> runRandBatch(RAND_W20, :chordalsdp)  # Chordal-LipSDP
  julia> runRandBatch(RAND_W10, :fastlip)     # Naive-Lip and CP-Lip
```
Each `runRandBatch(...)` call may take a while to run, and a CSV file is written to `dump/rand` after each network in the batch is finished.


## Running individual NNet files

Start up the Julia REPL and load a path to the NNet file
```
  julia -i scripts/run_nnet --nnet /path/to/file.nnet --tau 1
```
And once the REPL has started, there will be some predefined variables, and you may run, for instance
```
  julia> soln, lipconst = solveLipschitz(ffnet, weight_scales, :chordalsdp)
```
The `soln` object is a `QuerySolution` as defined in `src/Methods/header.jl`. The `lipconst` is a `Float64` corresponding to the Lipschitz constant. The accepted methods are `:lipsdp`, `:chordalsdp`, `:naivelp`, `:cplip`.

You may also customize queries, for instance:

```
  julia> my_mosek_opts = Dict("INTPNT_CO_TOL_DFEAS" => 1e-7)
  julia> my_chordalsdp_opts = ChordalSdpOptions(τ=3, mosek_opts=my_mosek_opts)
  julia> soln = solveLipschitz(ffnet, my_chordalsdp_opts)
```

However, you then need to apply your own scaling to get the correct Lipschitz constant, c.f. `src/FastNDeepLipSdp.jl`.


