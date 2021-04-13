include("hamiltonian.jl")

struct MCState
    spin_config::AbstractVector{Int}
    neighbours::Array{Array{Int64,1},1}
end

function MCState(H::AbstractIsing; init="rand")
    Ns = nspins(H)
    if init == "up"
        spin_config = ones(Ns)
    elseif init == "down"
        spin_config = zeros(Ns)
    else
        spin_config = rand(0:1, Ns)
    end

    neighbours = nearest_neighbours(H.dims)

    return MCState(spin_config, neighbours)
end

function nearest_neighbours(dim::Int)
    bonds = [Int[] for _ in 1:dim]
    for i in 2:dim-1
        push!(bonds[i], i+1)
        push!(bonds[i], i-1)
    end

    push!(bonds[1], 2)
    push!(bonds[dim], dim-1)
    push!(bonds[1], dim)
    push!(bonds[dim], 1)

    return bonds
end

function nearest_neighbours(dims::NTuple{2,Int})
    # order: upper, lower, right, left
    N = prod(dims)
    bonds = [Int[] for _ in 1:N]
    for i in 1:N
        push!(bonds[i], upper_neighbour_periodic(dims, i))
        push!(bonds[i], lower_neighbour_periodic(dims, i))
        push!(bonds[i], right_neighbour_periodic(dims, i))
        push!(bonds[i], left_neighbour_periodic(dims, i))
    end
    return bonds
end

to_pm1(spins::Vector{Int}) = 2 .* spins .- 1
to_pm1(spin::Int) = 2 * spin - 1    