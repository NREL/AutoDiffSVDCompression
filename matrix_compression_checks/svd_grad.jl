import Pkg
Pkg.activate(".")
Pkg.status()

import ForwardDiff as FD
import HeatEquation as HE
import ImplicitAD as IAD
import LinearAlgebra as LA
import ProgressMeter as PM
import ReverseDiff as RD

function HE.convert_kappa(::Type{T}, kappa::Real) where {T<:Real}
    return kappa
end

function HE.build_2d_heat_csc(::Type{T}, kappa, dt, Nx, Ny) where {T<:Real}
    return HE.build_2d_heat_csc(Float64, kappa, dt, Nx, Ny)
end

function HE.gridpoint(dim::I, k::I, N::I, ::Type{T}) where {I<:Integer,T<:RD.TrackedReal}
    return HE.gridpoint(dim, k, N, RD.valtype(T))
end

function HE.gridpoint(dim::I, k::I, N::I, ::Type{T}) where {I<:Integer,T<:FD.Dual}
    return HE.gridpoint(dim, k, N, FD.valtype(T))
end

function HE.my_linear_solve!(
    u_sol::AbstractVector,
    A::Any,
    A_fact,
    b::AbstractVector,
)
    u_sol .= IAD.implicit_linear(A, b; Af=A_fact)
    return u_sol
end

function my_source(tk::T, xi::T, yj::T, a::S, b::S, r::S, h::S) where {T,S}
    # return T((xi - a)^2 + (yj - b)^2 <= r^2)
    # val = T((xi - a)^2 + (yj - b)^2 <= r^2)
    dr = sqrt((xi - a)^2 + (yj - b)^2)
    # h = 5.0
    val = max(0.0, h * (1.0 - dr / r))
    # val = exp(0.5*(-(xi - a)^2 - (yj - b)^2) / r^2)
    return val
end

function eval_space_l2(prob::HE.HeatProblem, k::Integer)

    kappa = prob.kappa
    f = prob.f

    dt = prob.dt
    # Nt = prob.Nt
    Nx = prob.Nx
    Ny = prob.Ny

    A = prob.A
    A_fact = prob.A_fact
    uk = prob.uk
    rhs = prob.rhs

    HE.heat_step(A, A_fact, uk, rhs, kappa, f, k, dt, Nx, Ny)

    dx = HE.gridsize(1, Nx, Float64)
    dy = HE.gridsize(2, Ny, Float64)
    l2_norm = dx * dy * LA.norm(uk, 2)^2

    return l2_norm

end

function eval_spacetime_l2(
    prob::HE.HeatProblem;
    progress::Bool=false,
)

    k = 0
    dt = prob.dt
    Nt = prob.Nt

    if progress
        pm = PM.Progress(Nt)
    end

    # l2_norm = dt*dx*dy*LA.norm(uk, 2)^2
    l2_norm = 0.0 # Assumes u0 = zeros(Nx,Ny)

    while k < Nt
        l2_norm += dt * eval_space_l2(prob, k)
        k += 1
        if progress
            PM.next!(pm)
        end
    end

    if progress
        PM.finish!(pm)
    end

    return l2_norm

end

# function simulate_heat(x, p)
#     (a, b, r, h) = x
#     f(t, x, y) = my_source(t, x, y, a, b, r, h)
#     (kappa, tf, dt, N) = p
#     u0 = zeros(eltype(x), N, N)
#     if eltype(x) <: AbstractFloat
#         prob = HE.heat_setup_cpu(u0, kappa, f, tf, dt, N, N, -1, :csc)
#     elseif eltype(x) <: FD.Dual
#         prob = heat_setup_fd_cpu(u0, kappa, f, tf, dt, N, N, -1, :csc)
#     else
#         prob = heat_setup_rd_cpu(u0, kappa, f, tf, dt, N, N, -1, :csc)
#         # prob = HE.heat_setup_cpu(u0, kappa, f, tf, dt, N, N, -1, :csc)
#     end
#     HE.heat_loop(prob, nothing; progress=false)
#     return prob.uk
# end

function heat_setup_fd_cpu(
    u0::Matrix,
    kappa,
    interior::Function,
    tf::R,
    dt::R,
    Nx::I,
    Ny::I,
    save_rate::I,
    format::Symbol,
) where {I<:Integer, R<:Real}

    T = FD.valtype(eltype(u0))
    tf = convert(T, tf)
    dt = convert(T, dt)
    kappa = HE.convert_kappa(T, kappa)
    save_rate = convert(I, save_rate)

    return HE.heat_setup(u0, kappa, interior, tf, dt, Nx, Ny, save_rate, format)

end

function heat_setup_rd_cpu(
    u0::Matrix,
    kappa,
    interior::Function,
    tf::R,
    dt::R,
    Nx::I,
    Ny::I,
    save_rate::I,
    format::Symbol,
) where {I<:Integer, R<:Real}

    T = RD.valtype(eltype(u0))
    tf = convert(T, tf)
    dt = convert(T, dt)
    kappa = HE.convert_kappa(T, kappa)
    save_rate = convert(I, save_rate)

    return HE.heat_setup(u0, kappa, interior, tf, dt, Nx, Ny, save_rate, format)

end

function setup_heat(x, p)
    (a, b, r, h) = x
    f(t, x, y) = my_source(t, x, y, a, b, r, h)
    (kappa, tf, dt, N) = p
    u0 = zeros(eltype(x), N, N)
    if eltype(x) <: AbstractFloat
        prob = HE.heat_setup_cpu(u0, kappa, f, tf, dt, N, N, -1, :csc)
    elseif eltype(x) <: FD.Dual
        prob = heat_setup_fd_cpu(u0, kappa, f, tf, dt, N, N, -1, :csc)
    else
        prob = heat_setup_rd_cpu(u0, kappa, f, tf, dt, N, N, -1, :csc)
        # prob = HE.heat_setup_cpu(u0, kappa, f, tf, dt, N, N, -1, :csc)
    end
    return prob
end

function cost_x(x, p)
    # return 0.5*LA.norm(x,2)^2 + 0.25*LA.norm(x,4)^4 - log(x[3]) - log(x[4])
    return 0.5*LA.norm(x,2)^2 - log(x[3]) - log(x[4])
end

function cost_u(x, p; progress=true)
    prob = setup_heat(x, p)
    return 0.5*eval_spacetime_l2(prob; progress=progress)
end

function cost_u_svd(x, p; progress=true)
    prob = setup_heat(x, p)
    return 0.5*eval_spacetime_l2_svd(prob; progress=progress)
end

function cost(cu, cx)
    a = -1e0
    b = 1e0
    return a*cu + b*cx
end

function obj(x, p; progress=false)
    cx = cost_x(x, p)
    cu = cost_u(x, p; progress=progress)
    return cost(cu, cx)
end

function obj_svd(x, p; progress=false)
    cx = cost_x(x, p)
    cu = cost_u_svd(x, p; progress=progress)
    return cost(cu, cx)
end

function main()
    N = 3
    dt = 5e-1
    # tf = 100*dt
    tf = 1.0
    kappa = 1.0
    # u0 = zeros(N,N)
    params = (kappa, tf, dt, N)
    x0 = [0.9, -0.2, 0.5, 1.0]

    g0 = RD.gradient(x -> obj(x, params), x0)
    # gtp = RD.GradientTape(x -> obj(x, params), x0)
    # g0 = RD.gradient!(gtp, x0)

    @show g0

    return

end
