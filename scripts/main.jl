PATH="../src/"

include(PATH*"hamiltonian.jl")
include(PATH*"mc_state.jl")
include(PATH*"updates.jl")
include(PATH*"measure.jl")

using DrWatson: savename

using ArgParse
using Random
using DelimitedFiles
using JLD2
using JSON
using FileIO

SCRATCH_PATH = "../examples/data/"

function init_mc(parsed_args)
    J = parsed_args["interaction"]
    B = parsed_args["field"]
    L = parsed_args["L"]
    beta = parsed_args["beta"]
    seed = parsed_args["seed"]
    savesteps = parsed_args["savesteps"]

    Random.seed!(seed)
    H = NNIsing(J, B, (L, L), (true, true))
    mc_state = MCState(H)

    MCS = parsed_args["measurements"]
    EQ_MCS = div(MCS, 10)
    skip = parsed_args["skip"]
    
    mc_opts = (MCS, EQ_MCS, skip, beta, savesteps)

    d = (L=L, beta=beta, seed=seed)
    sname = savename(d; digits = 4)

    return H, mc_state, mc_opts, sname
end

function run(parsed_args)
    H, mc_state, mc_opts, sname = init_mc(parsed_args)
    MCS, EQ_MCS, skip, beta, savesteps = mc_opts

    energies = zeros(Float64, MCS)
    energies_sqr = zeros(Float64, MCS)
    mags = zeros(Float64, MCS)
    mags_sqr = zeros(Float64, MCS)

    # equil
    for i in 1:EQ_MCS
        #sw_update!(H, mc_state, beta)
        #wolff_update!(H, mc_state, beta)
        spin_flip_update!(H, mc_state, beta)
    end

    for i in 1:MCS
        #sw_update!(H, mc_state, beta)
        #wolff_update!(H, mc_state, beta)
        spin_flip_update!(H, mc_state, beta)

        energies[i] = energy(H, mc_state)
        energies_sqr[i] = energies[i]^2
        mags[i] = magnetization(mc_state.spin_config)
        mags_sqr[i] = mags[i]^2

        for _ in 1:skip
            #sw_update!(H, mc_state, beta)
            #wolff_update!(H, mc_state, beta)
            spin_flip_update!(H, mc_state, beta)
        end
    end

    if savesteps
        open(SCRATCH_PATH*"energies_J=$(H.J)_B=$(H.B)_beta=$(beta)_dims=$(H.dims).txt", "w") do io
            writedlm(io, energies)
        end
        open(SCRATCH_PATH*"energies_sqr_J=$(H.J)_B=$(H.B)_beta=$(beta)_dims=$(H.dims).txt", "w") do io
            writedlm(io, energies_sqr)
        end
        open(SCRATCH_PATH*"magnetization_J=$(H.J)_B=$(H.B)_beta=$(beta)_dims=$(H.dims).txt", "w") do io
            writedlm(io, mags)
        end
        open(SCRATCH_PATH*"sqr_magnetization_J=$(H.J)_B=$(H.B)_beta=$(beta)_dims=$(H.dims).txt", "w") do io
            writedlm(io, mags_sqr)
        end
    
    else
        E = stats_dict(energies)
        E2 = stats_dict(energies_sqr)
        M = stats_dict(mags)
        M2 = stats_dict(mags_sqr)

        obs_estimates = Dict{Symbol, Dict}()
        obs_estimates[:energy] = E
        obs_estimates[:sqr_energy] = E2
        obs_estimates[:magnetization] = M
        obs_estimates[:sqr_magnetization] = M2

        mkpath(SCRATCH_PATH)
        path = joinpath(SCRATCH_PATH, sname)

        observables_file = path * "_observables.json"

        open(observables_file, "w") do io
            JSON.print(io,
                Dict([k => obs_estimates[k] for k in keys(obs_estimates)]),
                2)
        end
    end

end

s = ArgParseSettings()

@add_arg_table! s begin
    "L"
        help = "2D dimension (i.e. L x L)"
        required = true
        arg_type = Int
    "-J", "--interaction"
        help = "Strength of the interactions"
        arg_type = Float64
        default = 1.0
    "-B", "--field"
        help = "Strength of the field"
        arg_type = Float64
        default = 0.0
    "--seed"
        help = "random seed"
        arg_type = Int
        default = 1234
    "--measurements", "-n"
        help = "Number of samples to record"
        arg_type = Int
        default = 100_000
    "--skip", "-s"
        help = "Number of MC steps to perform between each measurement"
        arg_type = Int
        default = 0
    "--beta"
        help = "Inverse temperature"
        arg_type = Float64
        default = 1.

    "--savesteps"
        help = "Save things as a function of MC steps"
        action = :store_true
end

parsed_args = parse_args(ARGS, s)

run(parsed_args)