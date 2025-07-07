module BurgersEquation

import ProgressMeter as PM

include("utilities.jl")
include("numerical_flux.jl")

struct BurgersHistory{
    I<:Integer,
    T<:Real,
}
    save_rate::I
    u::Dict{I,Vector{T}}
    t::Dict{I,T}
end

function BurgersHistory(
    rate::I, ::Type{T},
) where {I<:Integer,T<:Real}
    u = Dict{I,Vector{T}}()
    t = Dict{I,T}()
    return BurgersHistory(rate, u, t)
end

struct BurgersProblem{
    I<:Integer,
    R1<:Real,R2<:Real,R3<:Real,
    V<:AbstractVector,
    F<:Function,
    FP<:Function,
    NF<:Function,
    H<:Union{Nothing,BurgersHistory},
}
    Nx::I
    Nt::I
    dt::R1
    dx::R2
    tf::R3
    u0::V
    uk::V
    v::V # Scratch space
    f::F # Assumed to be convex with minimum at 0
    fprime::FP # Spatial derivative of f
    num_flux::NF
    hist::H
end

function save_solution(
    bhist::BurgersHistory,
    u::AbstractVector,
    tstep::Integer,
    time::Real
)
    if bhist.save_rate > 0 && mod(tstep, bhist.save_rate) == 0
        bhist.u[tstep] = Vector(u)
        bhist.t[tstep] = time
    end
    return
end

function save_solution(::Any, ::Any, ::Any, ::Any)
    return
end

function burger_step(u0, u1, f, fprime, flux, dt, dx, Nx)

    umax = maximum(u0)
    ratio = dt / dx
    cfl = umax * ratio
    if cfl > 1.0
        @warn("CFL: ", cfl)
    end
    # Periodic boundary conditions
    Fp = flux(f, fprime, u0[1], u0[2], ratio)
    Fm = flux(f, fprime, u0[Nx], u0[1], ratio)
    u1[1] = u0[1] - ratio * (Fp - Fm)

    Fp = flux(f, fprime, u0[Nx], u0[1], ratio)
    Fm = flux(f, fprime, u0[Nx-1], u0[Nx], ratio)
    u1[Nx] = u0[Nx] - ratio * (Fp - Fm)

    # # Neumann boundary conditions
    # u1[1] = u0[1]
    # u1[Nx] = u0[Nx]

    # u1[1] = 1.0

    for i in 2:Nx-1
        Fp = flux(f, fprime, u0[i], u0[i+1], ratio)
        Fm = flux(f, fprime, u0[i-1], u0[i], ratio)
        u1[i] = u0[i] - ratio * (Fp - Fm)
    end

    return

end

function burger_loop(u0, u1, f, fprime, nflux, dt, Nt, dx, Nx;
    history=nothing, progress)

    if progress
        pm = PM.Progress(Nt)
    end

    count = 0

    while count < Nt

        burger_step(u0, u1, f, fprime, nflux, dt, dx, Nx)
        count += 1
        save_solution(history, u1, count, count * dt)

        if progress
            PM.next!(pm)
        end

        u0 .= u1

    end

    if progress
        PM.finish!(pm)
    end

    return

end

function burger_loop(prob::BurgersProblem; progress=false)
    prob.v .= prob.u0
    return burger_loop(
        prob.v,
        prob.uk,
        prob.f,
        prob.fprime,
        prob.num_flux,
        prob.dt,
        prob.Nt,
        prob.dx,
        prob.Nx;
        history=prob.hist,
        progress=progress,
    )
end

function solve(prob::BurgersProblem; progress=false)
    return burger_loop(prob; progress)
end

function setup(
    u0::AbstractVector,
    f::Function,
    fprime::Function,
    tf::Real,
    dt::Real,
    Nx::Integer,
    flux::Symbol=:lax_wendroff;
    save_rate=-1
)

    dx = gridsize(Nx)
    Nt = Int(ceil(tf / dt))

    nflux = fetch_numerical_flux(flux)

    hist = save_rate > 0 ? BurgersHistory(save_rate, eltype(u0)) : nothing

    save_solution(hist, u0, 0, zero(eltype(u0)))

    prob = BurgersProblem(Nx, Nt, dt, dx, tf, u0, copy(u0), copy(u0),
        f, fprime, nflux, hist)

    return prob

end

function run(
    u0::AbstractVector,
    f::Function,
    fprime::Function,
    tf::Real,
    dt::Real,
    Nx::Integer,
    flux::Symbol=:lax_wendroff;
    progress=false,
    save_rate=-1,
)
    prob = setup(u0, f, fprime, tf, dt, Nx, flux; save_rate)
    burger_loop(prob; progress)
    return
end

end # module BurgersEquation
