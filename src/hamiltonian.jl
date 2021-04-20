# 2D Ising PBC

abstract type AbstractIsing end

# TODO: general Ising
# TODO: make sure dims = (x, 1) is interpreted as just dims = x (1D chain)
IntOrTupleInt = Union{Int, NTuple{2,Int}}
BoolOrTupleBool = Union{Int, NTuple{2,Bool}}

struct NNIsing <: AbstractIsing
    J::Float64
    B::Float64
    dims::IntOrTupleInt
    #PBC::NTuple{2, Bool}
    NNIsing(J,B,dims) = new(J,B,dims)
end

#=
struct GeneralIsing <: AbstractIsing
    J::UpperTriangular{Float64}
    B::Float64
    dims::IntOrTupleInt
    PBC::NTuple{Bool, 2}
    GeneralIsing(J,B,dims) = new(J,B,dims)
end
=#

nspins(H::AbstractIsing) = prod(H.dims)