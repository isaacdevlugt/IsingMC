include("hamiltonian.jl")
include("mc_state.jl")
include("updates.jl")
include("measure.jl")

using DrWatson: savename

using ArgParse
using Random
using DelimitedFiles
using JLD2
using JSON
using FileIO

SCRATCH_PATH = "../UWGrad/PHYS705/Project/"

measurementtodict(M::Measurement) = Dict("value" => M["mean"], "error" => M["error"], "variance" => M["variance"])

function init_mc(parsed_args)
    J = parsed_args["interaction"]
    B = parsed_args["field"]
    L = parsed_args["L"]
    beta = parsed_args["beta"]
    seed = parsed_args["seed"]

    Random.seed!(seed)
    H = NNIsing(J, B, (L, L), (false, false))
    mc_state = MCState(H)

    MCS = parsed_args["measurements"]
    EQ_MCS = div(MCS, 10)
    skip = parsed_args["skip"]
    
    mc_opts = (MCS, EQ_MCS, skip, beta)

    d = (L=L, beta=beta, seed=seed)
    sname = savename(d; digits = 4)

    return H, mc_state, mc_opts, sname
end


function partA(parsed_args)
    H, mc_state, mc_opts, sname = init_mc(parsed_args)
    MCS, EQ_MCS, skip, beta = mc_opts
    EQ_MCS = 0 # hard set for this question

    energies = zeros(Float64, MCS)

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

        for _ in 1:skip
            #sw_update!(H, mc_state, beta)
            #wolff_update!(H, mc_state, beta)
            spin_flip_update!(H, mc_state, beta)
        end
    end

    open("../UWGrad/PHYS705/Project/PartA/energies_J=$(H.J)_B=$(H.B)_beta=$(beta)_dims=$(H.dims).txt", "w") do io
        writedlm(io, energies)
    end
end


function partB(parsed_args)
    H, mc_state, mc_opts, sname = init_mc(parsed_args)
    MCS, EQ_MCS, skip, beta = mc_opts

    mags = zeros(Float64, MCS)
    energies = zeros(Float64, MCS)

    println("Running $(typeof(H)), beta = $(beta), dims = $(H.dims)",)
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
        mags[i] = magnetization(mc_state.spin_config)
        energies[i] = energy(H, mc_state)

        for _ in 1:skip
            #sw_update!(H, mc_state, beta)
            #wolff_update!(H, mc_state, beta)
            spin_flip_update!(H, mc_state, beta)
        end
    end

    E = stats_dict(energies)
    M = stats_dict(mags)

    obs_estimates = Dict{Symbol, Dict}()
    obs_estimates[:energy] = E
    obs_estimates[:magnetization] = M

    path = joinpath(SCRATCH_PATH, "PartB")
    mkpath(path)
    path = joinpath(path, sname)

    observables_file = path * "_observables.json"

    open(observables_file, "w") do io
        JSON.print(io,
            Dict([k => measurementtodict(obs_estimates[k]) for k in keys(obs_estimates)]),
            2)
    end

end


function partC(parsed_args)
    H, mc_state, mc_opts, sname = init_mc(parsed_args)
    MCS, EQ_MCS, skip, beta = mc_opts

    energies = zeros(Float64, MCS)
    sqr_energies = zeros(Float64, MCS)

    println("Running $(typeof(H)), beta = $(beta), dims = $(H.dims)",)
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
        #sqr_energies[i] = energies[i]^2

        for _ in 1:skip
            #sw_update!(H, mc_state, beta)
            #wolff_update!(H, mc_state, beta)
            spin_flip_update!(H, mc_state, beta)
        end
    end

    E = stats_dict(energies)

    path = joinpath(SCRATCH_PATH, "PartC")
    mkpath(path)
    path = joinpath(path, sname)

    observables_file = path * "_observables.json"

    open(observables_file, "w") do io
        JSON.print(io, Dict("energy" => Dict(k => E[k] for k in keys(E))))
    end
end

function partD(parsed_args)
    H, mc_state, mc_opts, sname = init_mc(parsed_args)
    MCS, EQ_MCS, skip, beta = mc_opts

    mags = zeros(Float64, MCS)
    sqr_mags = zeros(Float64, MCS)

    println("Running $(typeof(H)), beta = $(beta), dims = $(H.dims)",)
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
        mags[i] = magnetization(mc_state.spin_config)
        #sqr_mags[i] = mags[i]^2

        for _ in 1:skip
            #sw_update!(H, mc_state, beta)
            #wolff_update!(H, mc_state, beta)
            spin_flip_update!(H, mc_state, beta)
        end
    end

    M = stats_dict(mags)

    path = joinpath(SCRATCH_PATH, "PartD")
    mkpath(path)
    path = joinpath(path, sname)

    observables_file = path * "_observables.json"

    open(observables_file, "w") do io
        JSON.print(io, Dict("magnetization" => Dict(k => M[k] for k in keys(M))))
    end
end


s = ArgParseSettings()

@add_arg_table! s begin
    "partA"
        help = "Part A of project"
        action = :command
    "partB"
        help = "Part B of project"
        action = :command
    "partC"
        help = "Part C of project"
        action = :command
    "partD"
        help = "Part D of project"
        action = :command
end


@add_arg_table! s["partA"] begin
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
end

import_settings!(s["partB"], s["partA"])
import_settings!(s["partC"], s["partA"])
import_settings!(s["partD"], s["partA"])

parsed_args = parse_args(ARGS, s)

if parsed_args["%COMMAND%"] == "partA"
    partA(parsed_args["partA"])
elseif parsed_args["%COMMAND%"] == "partB"
    partB(parsed_args["partB"])
elseif parsed_args["%COMMAND%"] == "partC"
    partC(parsed_args["partC"])
elseif parsed_args["%COMMAND%"] == "partD"
    partD(parsed_args["partD"])
end