# Monte Carlo simulations of the 2D classical Ising model with periodic boundaries.

To run this code, you must have the following dependencies.

- DrWatson, ArgParse, Random, DelimitedFiles, JSON, FileIO, Measurements, StatsBase

After adding these packages, navigate to `src/` and run `julia main.jl --help` for a list of parameters to specify to run the simulation. Please note that non-zero magnetic fields are currently not supported. Simulations will save a `.json` file with energy, squared energy, magnetization, magnetization squared, and absolute magnetization. If `--savesteps` is specified, a `.txt` file containing the observables measured at every measurement step will be generated in place of the `.json` file.
