
import HeatEquation as HE
import LinearAlgebra as LA
import SparseArrays as SA

using Plots

function my_source(tk::T, xi::T, yj::T, a::T, b::T, r::T) where T
    # return T((xi - a)^2 + (yj - b)^2 <= r^2)
    # val = exp(0.5*(-(xi - a)^2 - (yj - b)^2) / r^2)
    dr = sqrt((xi - a)^2 + (yj - b)^2)
    h = 10.0
    val = max(0.0, h*(1.0 - dr / r))
    return val
end

function main()

    p = plot(yscale=:log10, yticks=[10.0^k for k in -20:1:4], dpi=300, xlabel="Index", ylabel="Singular Value")
    q = plot(yscale=:log10, yticks=[10.0^k for k in -20:1:4], xlim=(0,40), dpi=300, xlabel="Index", ylabel="Singular Value")

    kmax = 11
    kmin = 4
    Nmax = 2^kmax

    x0 = [0.9, -0.2, 0.5]

    for k in kmax:-1:kmin

        N = 2^k
        println("**** N = $N ****")
        dt = 1e-2
        tf = 1.0
        kappa = 1.0
        # f(t, x, y) = my_source(t, x, y, 0.0, 0.0, 0.5)
        f(t, x, y) = my_source(t, x, y, x0...)

        u0 = zeros(N, N)

        @time usol = HE.run_heat_cpu(u0, kappa, tf, dt, N, N; f=f, save_rate=-1)

        @time u_svd = LA.svd(usol)

        # @show u_svd
        scatter!(p, u_svd.S, label="N=$N")
        # plot!(p, 1:Nmax, HE.gridsize(1, N, Float64)*ones(Nmax), label=nothing)

        scatter!(q, u_svd.S, label="N=$N")
        # plot!(q, 1:Nmax, HE.gridsize(1, N, Float64)*ones(Nmax), label=nothing)

    end

    png(p, "heat_solution_singular_values")
    png(q, "heat_solution_singular_values_zoom")

    return

end

main()
