function endpoints()
    return (0.0, 1.0)
end

function gridsize(N::Integer)
    (a, b) = endpoints()
    return (b - a) / N
end

function gridpoint(k::Integer, N::Integer)
    (a, _) = endpoints()
    return a + (k-1)*gridsize(N)
end

function fetch_numerical_flux(my_numerical_flux::Symbol)
    nflux = Symbol(lowercase(string(my_numerical_flux)))
    if (nflux == :lax_friedrichs || nflux == :laxfriedrichs || nflux == :lf)
        nf = lax_friedrichs
    elseif (nflux == :lax_wendroff || nflux == :laxwendroff || nflux == :lw)
        nf = lax_wendroff
    else
        error("Unrecognized numerical flux: ", my_numerical_flux)
    end
    return nf
end

function space_grid(N::Integer)
    (a, b) = endpoints()
    return range(a, b, N+1)
end

function expand_solution(u::AbstractVector)
    N = length(u)
    u_ex = similar(u, N + 1)
    u_ex[1:N] .= u
    u_ex[N+1] = u[1]
    return u_ex
end

@inline es(u) = expand_solution(u)
