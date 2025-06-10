# import HeatEquation as HE
import BurgersEquation as BE
import LinearAlgebra as LA
import SparseArrays as SA

using Plots

function ic_smooth(x, a, b)
    return b + a*sin(2*pi*x)
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

function ic_hat(x, a, b)
    return hat_initial_condition(x, 0.0, 1.0, a, b)
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

    f(u) = 0.5*u^2
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
    BE.solve(bp)

    return bp

end

function main()

    p = plot(yscale=:log10, yticks=[10.0^k for k in -20:1:4], dpi=300, xlabel="Index", ylabel="Singular Value")
    q = plot(yscale=:log10, yticks=[10.0^k for k in -20:1:4], xlim=(0,20), dpi=300, xlabel="Index", ylabel="Singular Value")

    kmax = 12
    kmin = 4
    Nmax = 2^kmax

    cfl = 0.85
    tf = 1.0

    # x0 = [0.9, -0.2, 0.5]
    x0 = [0.2, 1.0]

    my_params = Dict(
        :cfl => cfl,
        :tf => tf,
        :flux => :lf,
        :ic => ic_hat,
    )

    for k in kmax:-2:kmin

        Nsq = 2^k
        N = Int(sqrt(Nsq))
        println("**** N = $Nsq ****")

        my_params[:Nx] = Nsq
        @time bp = burger_solution(x0, my_params)
        @show N
        usol = reshape(bp.uk, N, N)

        @time u_svd = LA.svd(usol)

        # @show u_svd
        scatter!(p, u_svd.S, label="N=$Nsq")
        # plot!(p, 1:Nmax, HE.gridsize(1, N, Float64)*ones(Nmax), label=nothing)

        scatter!(q, u_svd.S, label="N=$Nsq")
        # plot!(q, 1:Nmax, HE.gridsize(1, N, Float64)*ones(Nmax), label=nothing)

    end

    png(p, "burgers_$(string(my_params[:ic]))_singular_values")
    png(q, "burgers_$(string(my_params[:ic]))_singular_values_zoom")

    return

end

main()
