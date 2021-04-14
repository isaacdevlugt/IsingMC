using Measurements
using StatsBase

function magnetization(spin_config::AbstractVector{Int})
    return mean(x -> 2*x - 1, spin_config)
end

function stats_dict(f::Function, x::Vector{Float64})
    y = f.(x)
    μ, stdev = mean_and_std(y)
    σ = stdev / sqrt(length(x))
    _, var = mean_and_var(x)
    #μ = mean(y)
    #σ = std(y; mean=μ) / sqrt(length(x))
    dict = Dict("mean" => μ, "error" => σ, "variance" => var)
    return dict
end

function energy(H::NNIsing, mc_state::MCState)
    spin_config = to_pm1(mc_state.spin_config)
    neighbours = mc_state.neighbours
    Ns = nspins(H)
    z = H.dims isa NTuple ? 4. : 2. # coordination number
    
    E = 0.
    for i in 1:Ns
        s = spin_config[i]
        s_NNs = spin_config[neighbours[i]]
        E += H.B*s - H.J*sum(s_NNs)*s / z
    end

    return E / Ns
end

#=
function bootstrap(f::Function, x::vector)
=#

stats_dict(x::Vector{Float64}) = stats_dict(identity, x)