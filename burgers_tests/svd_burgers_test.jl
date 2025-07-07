import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.status()

import BurgersEquation as BE
import LinearAlgebra as LA
import SparseArrays as SA

using Plots

function ic_smooth(x, a, b)
    return b + a * sin(2 * pi * x)
end

function hat_initial_condition(x, hl, hr, hh, hb)
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

function ic_cliff(x, ::Any, ::Any)
    return x < 0.5 ? 0.2 * x + 0.5 : 0.2 * x + 0.3
end

function ic_hat(x, a, b)
    return hat_initial_condition(x, 0.0, 1.0, a, b)
end

function ic_weierstrass(x, a, b; N=5)
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

function initial_condition(x, p)

    Nx = p[:Nx]
    ic = p[:ic]
    u0 = zeros(eltype(x), Nx)

    for i in 1:Nx
        xi = BE.gridpoint(i, Nx)
        u0[i] = ic(xi, x...)
    end

    return u0

end

function burger_solution(x, p; save=false)

    f(u) = 0.5 * u^2
    fu(u) = u

    u0 = initial_condition(x, p)
    Nx = p[:Nx]
    cfl = p[:cfl]
    # Assumes u >= 0
    umax = maximum(u0)
    dx = BE.gridsize(Nx)
    dt = cfl * dx / umax

    save_rate = save ? Int(floor(2e-3 / dt)) : -1

    bp = BE.setup(u0, f, fu, p[:tf], dt, Nx, p[:flux]; save_rate=save_rate)
    BE.solve(bp; progress=true)

    return bp

end

function svd_dimensions(ngrid)

    k = log2(ngrid)
    m = 2^Int(ceil(k / 2))
    n = 2^Int(floor(k / 2))

    return (m, n)

end

function main()

    for ic in [ic_cliff, ic_hat, ic_smooth, ic_weierstrass]

        println("######## Initial Condition: $(ic) ########")

        p0 = plot(yscale=:log10, yticks=[10.0^k for k in -20:1:4], dpi=300, xlabel="Index", ylabel="Singular Value")
        q0 = plot(yscale=:log10, yticks=[10.0^k for k in -20:1:4], xlim=(0, 40), dpi=300, xlabel="Index", ylabel="Singular Value")
        p1 = plot(yscale=:log10, yticks=[10.0^k for k in -20:1:4], dpi=300, xlabel="Index", ylabel="Singular Value")
        q1 = plot(yscale=:log10, yticks=[10.0^k for k in -20:1:4], xlim=(0, 40), dpi=300, xlabel="Index", ylabel="Singular Value")

        kmax = 14
        kmin = 4
        Nmax = 2^kmax

        cfl = 0.85
        tf = 1.0
        Tint = Int(tf)

        # x0 = [0.9, -0.2, 0.5]
        # x0 = [0.2, 1.0]

        my_params = Dict(
            :cfl => cfl,
            :tf => tf,
            :flux => :lf,
            :ic => ic,
        )

        if my_params[:ic] == ic_weierstrass
            a = 0.825
            b = 7
            x0 = [a, b]
        else
            x0 = [0.2, 1.0]
        end

        for k in kmax:-1:kmin

            N = 2^k
            println("**** N = $N ****")

            my_params[:Nx] = N
            @time bp = burger_solution(x0, my_params)
            # @show N
            usol = reshape(bp.uk, svd_dimensions(N))
            @time u_svd1 = LA.svd(usol)
            usol = reshape(bp.u0, svd_dimensions(N))
            @time u_svd0 = LA.svd(usol)

            # @show u_svd
            scatter!(p0, u_svd0.S, label="N=$N")
            scatter!(p1, u_svd1.S, label="N=$N")
            # plot!(p, 1:Nmax, HE.gridsize(1, N, Float64)*ones(Nmax), label=nothing)

            scatter!(q0, u_svd0.S, label="N=$N")
            scatter!(q1, u_svd1.S, label="N=$N")
            # plot!(q, 1:Nmax, HE.gridsize(1, N, Float64)*ones(Nmax), label=nothing)

            if k == kmax
                p = plot(BE.space_grid(N), BE.expand_solution(bp.u0); label="Initial", dpi=300,)
                plot!(p, BE.space_grid(N), BE.expand_solution(bp.uk); label="Final")
                png(p, "burgers_$(string(my_params[:ic]))")
            end

        end

        png(p0, "burgers_$(string(my_params[:ic]))_T0_singular_values")
        png(q0, "burgers_$(string(my_params[:ic]))_T0_singular_values_zoom")
        png(p1, "burgers_$(string(my_params[:ic]))_T$(Tint)_singular_values")
        png(q1, "burgers_$(string(my_params[:ic]))_T$(Tint)_singular_values_zoom")

    end

    return

end

main()
