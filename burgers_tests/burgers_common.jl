
import Distributed
import Random
import ReverseDiff

import ImplicitAD as IAD
import LinearAlgebra as LA
import ReverseDiff as RD

using BurgersEquation
import BurgersEquation as BE

using Plots

function make_gif(bp::BurgersEquation.BurgersProblem, gif_name; fps=20)

    x_grid = BurgersEquation.space_grid(bp.Nx)
    hist = bp.hist

    umin = floor(minimum(hist.u[0]); digits=1)
    umax = ceil(maximum(hist.u[0]); digits=1)

    ani = @animate for k in sort(collect(keys(hist.t)))
        tu = hist.t[k]
        uk = hist.u[k]
        plot(
            x_grid, BE.expand_solution(uk),
            title="Time: $(tu)", legend=false, ylim=(umin, umax)
        )
    end

    return gif(ani, gif_name * ".gif", fps=fps)

end

function implicit_burger_step(u0, params)

    prob = params[:prob]
    u1 = similar(u0)
    f = prob.f
    fprime = prob.fprime
    nflux = prob.num_flux
    dt = prob.dt
    dx = prob.dx
    Nx = prob.Nx

    BurgersEquation.burger_step(u0, u1, f, fprime, nflux, dt, dx, Nx)

    return u1

end

function residual_burger_step(r, u1, u0, params)

    prob = params[:prob]
    f = prob.f
    fprime = prob.fprime
    flux = prob.num_flux
    dt = prob.dt
    Nt = prob.Nt
    dx = prob.dx
    Nx = prob.Nx
    ratio = dt / dx

    # Periodic boundary conditions
    Fp = flux(f, fprime, u0[1], u0[2], ratio)
    Fm = flux(f, fprime, u0[Nx], u0[1], ratio)
    r[1] = u1[1] - u0[1] + ratio * (Fp - Fm)

    Fp = flux(f, fprime, u0[Nx], u0[1], ratio)
    Fm = flux(f, fprime, u0[Nx-1], u0[Nx], ratio)
    r[Nx] = u1[Nx] - u0[Nx] + ratio * (Fp - Fm)

    for i in 2:Nx-1
        Fp = flux(f, fprime, u0[i], u0[i+1], ratio)
        Fm = flux(f, fprime, u0[i-1], u0[i], ratio)
        r[i] = u1[i] - u0[i] + ratio * (Fp - Fm)
    end

    return

end

function residual_jacobian_y(r, u1, u0, p)
    return LA.I
end

function my_burger_loop(
    prob::BurgersEquation.BurgersProblem, p;
    progress::Bool=false, mode::Symbol=:implicit
)

    prob.v .= prob.u0
    u0 = prob.v
    dt = prob.dt
    Nt = prob.Nt
    hist = prob.hist

    if progress
        pm = PM.Progress(Nt)
    end

    count = 0

    while count < Nt

        if mode == :direct
            u1 = implicit_burger_step(u0, p)
        elseif mode == :implicit
            u1 = IAD.implicit(
                implicit_burger_step, residual_burger_step, u0, p;
                drdy=residual_jacobian_y
            )
        elseif mode == :svd
            u1 = IAD.implicit_svd(
                implicit_burger_step, residual_burger_step, u0, p;
                drdy=residual_jacobian_y
            )
        else
            error("Unrecognized mode: $mode")
        end

        count += 1
        ReverseDiff.@skip(BurgersEquation.save_solution(hist, u1, count, count * dt))

        if progress
            PM.next!(pm)
        end

        u0 = u1

    end

    prob.uk .= u0

    if progress
        PM.finish!(pm)
    end

    return

end

function set_dt_nsteps(tf::Real, umax::Real, cfl::Real, Nx::Integer)
    dx = BurgersEquation.gridsize(Nx)
    umax = max(umax, 1e-12)
    dt = 1.0 / ceil(tf * umax / (cfl * dx))
    Nt = Int(round(tf / dt))
    return (dt, Nt)
end

function burger_setup(x, p; save::Bool=false)

    f(u) = 0.5 * u^2
    fu(u) = u

    u0 = initial_condition(x, p)
    # @show eltype(u0)
    Nx = p[:Nx]
    cfl = p[:cfl]
    tf = p[:tf]
    # Assumes u >= 0
    umax = maximum(u0)
    dx = BurgersEquation.gridsize(Nx)
    # dt = cfl * dx / umax
    # dt = 1.0 / ceil(Tf / dt)
    # dt = 1.0 / ceil(tf * umax / (cfl * dx))
    (dt, Nt) = set_dt_nsteps(tf, umax, cfl, Nx)

    save_rate = save ? max(1, Int(floor(2e-3 / dt))) : -1
    cfl = dt * umax / dx

    # if p[:mode] == :svd
    #     tol = get(p, :tol, 0.0)
    #     nsv = get(p, :nsv, 3)
    #     println("SVD Mode -- " * (tol > 0.0 ? "tol: $tol" : "nsv: $nsv"))
    # end

    bp = BurgersEquation.setup(
        u0, f, fu, p[:tf], Nt, Nx, p[:flux];
        save_rate=save_rate
    )

    @assert(bp.Nt == Nt)

    return bp

end

function burger_solution(
    x,
    p;
    save::Bool=false,
    progress::Bool=false,
    mode::Symbol=:normal
)

    bp = burger_setup(x, p; save=save)

    # @show (bp.Nx, bp.Nt, bp.dx, bp.dt)

    p[:prob] = bp
    if mode == :normal
        BurgersEquation.solve(bp; progress=progress)
    else
        my_burger_loop(bp, p; progress=progress, mode=mode)
    end

    return bp

end

# function ic_smooth(x, a, b)
#     return b + a * sin(2 * pi * x)
# end

# function ic_hat(x, a, b)
#     return hat_initial_condition(x, 0.0, 1.0, b, a)
# end

function hat_function(x, hl, hr, hh, hb)
    hm = 0.5 * (hr + hl)
    if hl < x && x < hm
        v = (hh - hb) / (hm - hl) * (x - hl) + hb
    elseif hm <= x && x < hr
        v = (hb - hh) / (hr - hm) * (x - hr) + hb
    else
        v = hb * one(x)
    end
    return v
end

function grid_control(i, xi, x, p)
    return x[i]
end

function fourier_control(i, xi, x, p)
    # TODO: Implement...
    return
end

function tf_flat(x)
    return 1.0
end

function tf_sin(x)
    return 0.5 + 0.2 * sin(2 * pi * x)
end

function tf_cliff(x)
    # return x < 0.5 ? 0.6 : 0.4
    return x < 0.5 ? 0.2 * x + 0.5 : 0.2 * x + 0.3
end

function ic_weierstrass(x, a, b, N)
    vmax = 0.9
    vmin = 0.1
    vmid = 0.5 * (vmax + vmin)

    val = 0.0
    wbd = 0.0
    for n in 1:N
        val += a^n * cos(b^n * pi * x)
        wbd += a^n
    end

    val = 0.5 * (val + wbd) * (vmax - vmin) / wbd + vmin

    return val
end

function tf_weierstrass(x, params; a=0.825, b=7, N=5)

    Nx = params[:Nx]
    u0 = zeros(eltype(x), Nx)

    # sparams = copy(params)
    # sparams[:Nx] = 2^14

    for i in 1:Nx
        xi = BurgersEquation.gridpoint(i, Nx)
        u0[i] = ic_weierstrass(xi, a, b, N)
    end

    return u0

    # bp = burger_solution(u0, params)

    # return bp.uk

end

function initial_condition(x, p)

    Nx = p[:Nx]
    ic = p[:ic]
    u0 = zeros(eltype(x), Nx)

    for i in 1:Nx
        xi = BurgersEquation.gridpoint(i, Nx)
        u0[i] = ic(i, xi, x, p)
    end

    return u0

end

function target_condition(x, p)

    Nx = p[:Nx]
    target = p[:target]

    if (target == tf_weierstrass)
        return tf_weierstrass(x, p)
    end

    utar = zeros(eltype(x), Nx)

    for i in 1:Nx
        xi = BurgersEquation.gridpoint(i, Nx)
        utar[i] = target(xi)
    end

    return utar

end

function cost_u(x, p)
    bp = burger_solution(x, p; save=false, mode=p[:mode])
    # utar = target_condition(x, p)
    utar = p[:target]
    dx = BurgersEquation.gridsize(p[:Nx])
    # return (bp.uk[end] - 1.0)^2
    return 0.5 * dx * LA.norm(bp.uk - utar)^2
end

function cost_x(x, p)
    # return 0.5 * LA.norm(x, 2)^2
    Nx = p[:Nx]
    dx = BurgersEquation.gridsize(p[:Nx])
    val = 0.0
    for i in 1:Nx
        xi = x[i]
        val += 0.5 * dx * (xi - 0.5)^2
    end
    return val
    # return 0.5 * dx * LA.norm(x, 2)^2
end

function cost(x, p)
    a = get(p, :a, 10.0)
    b = get(p, :b, 10.0)
    scale = p[:scale]
    return scale * (b * cost_u(x, p) + a * cost_x(x, p))
end

function svd_dimensions(ngrid)
    k = log2(ngrid)
    m = 2^Int(ceil(k / 2))
    n = 2^Int(floor(k / 2))
    return (m, n)
end

function make_case_string(Nx::Integer, target::Symbol)
    return "$(target)_gs$(Nx)"
end

function random_initial_point(x0, seed, Nx, umin, umax)
    rng = Random.MersenneTwister(seed)
    x0 .= (umax - umin) .* randn(rng, Nx) .+ 0.5 * (umax + umin)
    return x0
end

function gradient_descent_step(x0, gradf0, alpha)
    x0 .-= alpha .* gradf0
    return x0
end

struct MyResult
    idx::Int
    gres::Float64
    ginf::Float64
    gmag::Float64
    svdmag::Float64
    dp::Float64
end

function gradient_error_inner_loop(
    result_channel,
    seed, ndescent, Nx, fgrad, fgrad_svd, alpha,
    job,
)

    # error("STOP!!!")

    umin = 0.45
    umax = 0.55

    x0 = zeros(Nx)
    offset = (ndescent + 1) * (job - 1)
    # @show seed + offset
    random_initial_point(x0, seed + offset, Nx, umin, umax)
    gtrue = zeros(Nx)
    gsvd = zeros(Nx)
    gerr = zeros(Nx)

    for step in 0:ndescent

        my_idx = offset + step + 1

        fgrad(gtrue, x0)
        fgrad_svd(gsvd, x0)

        gerr .= gsvd .- gtrue

        gres = LA.norm(gerr, 2)
        ginf = LA.norm(gerr, Inf)
        gmag = LA.norm(gtrue, 2)
        svdmag = LA.norm(gsvd, 2)
        dp = LA.dot(gtrue, gsvd)

        my_result = MyResult(my_idx, gres, ginf, gmag, svdmag, dp)
        # println("Queueing result")
        put!(result_channel, my_result)

        gradient_descent_step(x0, gtrue, alpha)

    end

    return

end
