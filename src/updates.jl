# swendsen wang
# wolff
# single spin flip

using DataStructures
using Statistics
using DelimitedFiles

include("mc_state.jl")

function ΔE(H::NNIsing, mc_state::MCState, site::Int)
    # energy difference by flipping a site
    # TODO: general Jij expression
    # TODO: include magnetic field
    spin = to_pm1(mc_state.spin_config[site])
    NNspins = to_pm1(mc_state.spin_config[mc_state.neighbours[site]])
    return 2*H.J*spin*sum(NNspins)
end

function spin_flip_update!(H::NNIsing, mc_state::MCState, β::Float64)
    N = nspins(H)
    shuffled_sites = collect(1:N)[randperm(N)]
    spin_config = mc_state.spin_config

    for site in shuffled_sites
        site = rand(1:nspins(H))
        Ediff = ΔE(H, mc_state, site)
        spin_config = mc_state.spin_config

        if rand() < min(1, exp(-β*Ediff))
            spin_config[site] ⊻= 1
        end
    end
end

function wolff_update!(H::NNIsing, mc_state::MCState, β::Float64)
    # TODO: general ising 

    Ns = nspins(H)
    spin_config = mc_state.spin_config
    P_add = 1 - exp(-2*β*H.J)

    in_cluster = falses(Ns)
    attempt = Int[]
    cluster = Int[] # This is the stack of sites in a cluster

    seed_site = rand(1:Ns)
    in_cluster[seed_site] = true
    append!(cluster, seed_site)
    append!(attempt, seed_site)
    while !isempty(attempt)
        current_site = attempt[1]
        NNs = mc_state.neighbours[current_site]
        for (i,site) in enumerate(NNs)
            if !in_cluster[site] && (spin_config[site] == spin_config[seed_site] && rand() < P_add) # parallel spin check
                append!(cluster, site)
                in_cluster[site] = true
                # TODO: exclude the current_site from the new NNs to attempt
                #exclude_idx = round(Int, mod(i+2, 4.1)) # 1 <---> 3, 2 <---> 4
                #newNNs = select_nearest_neighbours(H, site, exclude_idx)
                append!(attempt, mc_state.neighbours[site])
            end
        end
        # remove the current_site from the attempts
        popfirst!(attempt) 
    end
    # flip the cluster
    spin_config[cluster] .⊻= 1
end


function sw_update!(H::NNIsing, mc_state::MCState, β::Float64)
    # TODO: general ising 

    Ns = nspins(H)
    spin_config = mc_state.spin_config
    P_add = 1 - exp(-2*β*H.J)

    in_cluster = falses(Ns)
    attempt = Int[]
    cluster = Int[] # This is the stack of sites in a cluster

    for seed_site in 1:Ns
        if !in_cluster[seed_site]
            in_cluster[seed_site] = true
            append!(cluster, seed_site)
            append!(attempt, seed_site)
        else
            continue
        end

        while !isempty(attempt)
            current_site = attempt[1]
            NNs = mc_state.neighbours[current_site]
            for (i,site) in enumerate(NNs)
                if !in_cluster[site] && (spin_config[site] == spin_config[seed_site] && rand() < P_add) # parallel spin check
                    append!(cluster, site)
                    in_cluster[site] = true
                    # TODO: exclude the current_site from the new NNs to attempt
                    #exclude_idx = round(Int, mod(i+2, 4.1)) # 1 <---> 3, 2 <---> 4
                    #newNNs = select_nearest_neighbours(H, site, exclude_idx)
                    append!(attempt, mc_state.neighbours[site])
                end
            end
            # remove the current_site from the attempts
            popfirst!(attempt) 
        end
        # flip the cluster
        if rand() < 0.5
            spin_config[cluster] .⊻= 1
        end
        empty!(cluster)
        empty!(attempt)
    end
end

upper_neighbour_periodic(dims::NTuple{2,Int}, site::Int) = site + dims[1] > prod(dims) ? mod(site + dims[1], prod(dims)) : site + dims[1]
#upper_neighbour(dims::NTuple{2,Int}, site::Int) = site + dims[1]

lower_neighbour_periodic(dims::NTuple{2,Int}, site::Int) = site <= dims[1] ? site + prod(dims) - dims[1] : site - dims[1]
#lower_neighbour(dims::NTuple{2,Int}, site::Int) = site - dims[1]

right_neighbour_periodic(dims::NTuple{2,Int}, site::Int) = mod(site, dims[1]) == 0 ? site - dims[1] + 1 : site + 1
#right_neighbour(dims::NTuple{2,Int}, site::Int) = site + 1

left_neighbour_periodic(dims::NTuple{2,Int}, site::Int) = mod(site-1, dims[1]) == 0 ? site + dims[1] - 1 : site - 1
#left_neighbour(dims::NTuple{2,Int}, site::Int) = site - 1