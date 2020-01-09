using Random
using Distributions

# Only for 2D

function spinflip(spins, idx)
    #s[idx[1], idx[2]] = mod((s[idx[1], idx[2]] + 1), 2)
    spins[idx[1], idx[2]] *= -1
    return spins
end

function energy_diff(L, J, s, idx)
    nn = s[mod1(idx[1] - 1, L), idx[2]] + s[mod1(idx[1] + 1, L), idx[2]] + s[idx[1], mod1(idx[2] - 1, L)] + s[idx[1], mod1(idx[2] + 1, L)]
    return 2*J*s[idx[1], idx[2]]*nn
end

function acceptance(beta, L, J, s, idx)
    ediff = energy_diff(L, J, s, idx)
    
    if ediff <= 0
        return 1
    else
        return exp(-beta*ediff)
    end
end

function runMC(beta, J, L)
    s = rand(0:1, 100, 100)
    #s = zeros(100, 100)
    s = (s.*2) .- 1

    for t in 1:7e7
        idx = [rand(1:L) rand(1:L)]
        r = rand(Uniform(0,1))
        a = acceptance(beta, L, J, s, idx)
        
        if r <= a
            s = spinflip(s, idx)
        end

   end
   return s 
end

J = 1.0
k = 1.38e-23
beta = 1/2.0
L = 100

s = runMC(beta, J, L)
mag = sum(s)/(L^2)
println(mag)
