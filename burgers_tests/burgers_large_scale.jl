import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Pkg.status()

import ImplicitAD as IAD
import LinearAlgebra as LA
import ForwardDiff
import ForwardDiff as FD
import ReverseDiff
import ReverseDiff as RD

import ArgParse
import CSV
import Optim
import Printf
import Random
import Tables

using Plots

using BurgersEquation
import BurgersEquation as BE

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

function meminfo_julia(io::IOStream)
    GC.gc()
    # 2^20 = 1024^2 --> bytes to Megabytes
    l1 = Printf.@sprintf "GC live:   %9.3f MiB\n" Base.gc_live_bytes() / 2^20
    l2 = Printf.@sprintf "JIT:       %9.3f MiB\n" Base.jit_total_bytes() / 2^20
    l3 = Printf.@sprintf "Max. RSS:  %9.3f MiB\n" Sys.maxrss() / 2^20
    print(io, l1)
    print(io, l2)
    print(io, l3)
end

function dump_info(fname::AbstractString, time::Real, res)

    iof = open(fname, "w")
    meminfo_julia(iof)
    println(iof, "Seconds: ", time)

    if res !== nothing
        println(iof, "Number of Calls: ", Optim.f_calls(res))
        println(iof, "Converged: ", Optim.converged(res))
        println(iof, "Gradient Residual: ", res.g_residual)
        println(iof, "x Absolute Change: ", res.x_abschange)
    end

    close(iof)

    return

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

# # wrap residual function in a explicit form for convenience and
# # ensure type of r is appropriate
# function residual_wrap(yw, xw, pw)
#     T = promote_type(eltype(xw), eltype(yw))
#     rw = zeros(T, length(yw))  # match type of input variables
#     residual_burger_step(rw, yw, xw, pw)
#     return rw
# end

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
            # prob.uk .= u1
        elseif mode == :svd
            u1 = IAD.implicit_svd(
                implicit_burger_step, residual_burger_step, u0, p;
                drdy=residual_jacobian_y
            )
            # prob.uk .= u1
        else
            error("Unrecognized mode: $mode")
        end

        # ReverseDiff.@skip(prob.uk .= u1)

        count += 1
        ReverseDiff.@skip(BurgersEquation.save_solution(hist, u1, count, count * dt))

        if progress
            PM.next!(pm)
        end

        # @show eltype(u0)
        # @show eltype(u1)

        u0 = u1

    end

    prob.uk .= u0

    if progress
        PM.finish!(pm)
    end

    return

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
    dt = 1.0 / ceil(tf * umax / (cfl * dx))

    save_rate = save ? max(1, Int(floor(2e-3 / dt))) : -1
    cfl = dt * umax / dx

    # if p[:mode] == :svd
    #     tol = get(p, :tol, 0.0)
    #     nsv = get(p, :nsv, 3)
    #     println("SVD Mode -- " * (tol > 0.0 ? "tol: $tol" : "nsv: $nsv"))
    # end

    bp = BurgersEquation.setup(
        u0, f, fu, p[:tf], dt, Nx, p[:flux];
        save_rate=save_rate
    )

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

    bp = burger_solution(u0, params)

    return bp.uk

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
    a = 10.0
    b = 10.0
    scale = p[:scale]
    return scale * (b * cost_u(x, p) + a * cost_x(x, p))
end

function svd_dimensions(ngrid)
    k = log2(ngrid)
    m = 2^Int(ceil(k / 2))
    n = 2^Int(floor(k / 2))
    return (m, n)
end

function run_optimization(case, x0, f, fgrad, my_options)

    runtime = @elapsed begin

        if case == :forward

            # Optimization with FD
            println("-------- Running Optimimzation with ForwardDiff --------")
            @time res = Optim.optimize(
                f,
                x0,
                Optim.BFGS(),
                my_options;
                autodiff=:forward, # uses ForwardDiff.jl
            )

        elseif case == :finitediff

            # Optimization with finite differences
            println("-------- Running Optimimzation with Finite Diff --------")
            @time res = Optim.optimize(
                f,
                x0,
                Optim.BFGS(),
                my_options;
            )

        elseif case == :reverse || case == :impreverse || case == :dirreverse

            # Optimization with RD
            println("-------- Running Optimimzation with ReverseDiff --------")
            @time res = Optim.optimize(
                f,
                fgrad,
                x0,
                Optim.BFGS(),
                my_options;
            )

            # elseif case == :svdforward

            #     # Optimization with SVD and FD
            #     println("-------- Running Optimimzation with SVD and ForwardDiff --------")
            #     @time res = Optim.optimize(
            #         svd_objective,
            #         # svd_gradient,
            #         # lb,
            #         # ub,
            #         x0,
            #         # Optim.Fminbox(Optim.BFGS()),
            #         Optim.BFGS(),
            #         my_options;
            #         autodiff=:forward, # uses ForwardDiff.jl
            #     )

        elseif case == :svdreverse

            # Optimization with SVD and RD
            println("-------- Running Optimimzation with SVD and ReverseDiff --------")
            @time res = Optim.optimize(
                f,
                fgrad,
                x0,
                Optim.BFGS(),
                my_options;
            )

        else

            error("Unrecognized execution mode: $(case)")

        end
    end

    @show Optim.converged(res)
    @show Optim.minimum(res)
    @show res
    xsol = Optim.minimizer(res)

    return runtime, xsol, res

end

function finite_diff_gradient(g0, f, x0)

    # Done has described in Optim.jl documentation
    dx = eps(eltype(x0))^(1 / 3)
    xp = similar(x0)

    for k in eachindex(x0)
        # xp .= x0
        copyto!(xp, x0)
        xp[k] += dx
        fp = f(xp)
        xp[k] = x0[k] - dx
        fm = f(xp)
        g0[k] = 0.5 * (fp - fm) / dx
    end

    return

end

function run_gradient(case, x0, g0, f, fgrad)

    runtime = @elapsed begin

        if case == :forward

            # Optimization with FD
            println("-------- Evaluating Gradient with ForwardDiff --------")
            # Force compilation
            @time ForwardDiff.gradient!(g0, f, x0)

        elseif case == :finitediff

            # Optimization with finite differences
            println("-------- Evaluating Gradient with Finite Diff --------")
            # dx = sqrt(eps)
            @time finite_diff_gradient(g0, f, x0)

        elseif case == :reverse || case == :impreverse || case == :dirreverse

            # Optimization with RD
            println("-------- Evaluating Gradient with ReverseDiff --------")
            @time fgrad(g0, x0)

        elseif case == :svdreverse

            # Optimization with SVD and RD
            println("-------- Evaluating Gradient with SVD and ReverseDiff --------")
            @time fgrad(g0, x0)

        else

            error("Unrecognized execution mode: $(case)")

        end

    end

    return (runtime, x0)

end

function main(ARGS)

    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--case"
        help = "automatic differentiation method to use during optimization"
        arg_type = String
        required = true
        "--gridsize"
        help = "size of the grid to use when solving Burgers equation -- 2^gridsize"
        arg_type = Int
        required = true
        "--seed"
        help = "seed to send to random number generator for optimization initial condition"
        arg_type = Int
        required = true
        "--target"
        help = "function giving the target of function of the optimization"
        arg_type = String
        required = true
        "--trace"
        help = "print the trace from the optimization"
        action = :store_true
        "--extended-trace"
        help = "print the extended trace from the optimization"
        action = :store_true
        "--optimization"
        help = "run optimization benchmark problem"
        action = :store_true
        "--gradient"
        help = "run single gradient evaluation"
        action = :store_true
        # "--opt2", "-o"
        # help = "another option with an argument"
        # arg_type = Int
        # default = 0
        # "--flag1"
        # help = "an option without argument, i.e. a flag"
        # action = :store_true
        # "arg1"
        # help = "a positional argument"
        # required = true
    end

    parsed_args = ArgParse.parse_args(ARGS, s)

    if parsed_args["optimization"] && parsed_args["gradient"]
        error("Both --optimization and --gradient flags have been set.")
    end

    optimization = (parsed_args["optimization"] || !parsed_args["gradient"])

    case = Symbol(lowercase(parsed_args["case"]))
    k = parsed_args["gridsize"]

    tf = 1.0
    Nx = 2^k
    cfl = 0.85
    # tar_func = tf_cliff
    # tar_func = tf_sin

    tarf = Symbol(parsed_args["target"])
    if (tarf == :sin)
        tar_func = tf_sin
    elseif (tarf == :cliff)
        tar_func = tf_cliff
    elseif (tarf == :weierstrass)
        tar_func = tf_weierstrass
    else
        error("Unknown target function: $(tarf)")
    end

    @show tf
    @show Nx
    @show tar_func

    # # Initial point for sin/smooth target function
    # x0 = collect(range(0.4, 0.6, Nx))
    # Initial point for cliff target function
    seed = parsed_args["seed"]
    @show seed
    rng = Random.MersenneTwister(seed)
    umax = 0.505
    umin = 0.495
    # x0 = fill(0.5, Nx)
    x0 = (umax - umin) .* randn(rng, Nx) .+ 0.5 * (umax + umin)

    (m, n) = svd_dimensions(Nx)
    @show (m, n)

    # | ∇f(x) | < tol terminates
    tol = 1e-4
    # | x1 - x0 | < xtol terminates
    xtol = 1e-8
    my_options = Optim.Options(
        g_abstol=tol,
        x_abstol=xtol,
        outer_g_abstol=tol,
        outer_x_abstol=xtol,
        store_trace=false,
        extended_trace=parsed_args["extended-trace"],
        show_trace=(parsed_args["trace"] | parsed_args["extended-trace"]),
    )

    my_params = Dict(
        :Nx => Nx,
        :cfl => cfl,
        :tf => tf,
        :flux => :lf,
        :scale => 1e2,
        :ic => grid_control,
        :target => tar_func,
        :mode => :normal,
    )

    # rvs_params = copy(my_params)
    # rvs_params[:mode] = :direct

    # svd_params = copy(my_params)
    # svd_params[:mode] = :svd
    # # svd_params[:nsv] = 8
    # svd_params[:tol] = 1e-5
    # svd_params[:forward_svd] = true
    # svd_params[:matdim] = (m, n)

    if case == :reverse || case == :impreverse || case == :dirreverse

        if case == :reverse
            my_params[:mode] = :normal
        elseif case == :dirreverse
            my_params[:mode] = :direct
        else
            my_params[:mode] = :implicit
        end

        # rvs_objective(x) = cost(x, my_params)
        function rvs_gradient(g, x)
            RD.gradient!(g, my_objective, x)
            return
        end

        my_gradient = rvs_gradient

    elseif case == :svdreverse

        my_params[:mode] = :svd
        my_params[:tol] = 1e-5
        my_params[:matdim] = (m, n)

        # my_objective(x) = cost(x, my_params)
        function svd_gradient(g, x)
            RD.gradient!(g, my_objective, x)
            return
        end

        my_gradient = svd_gradient

    else

        function no_gradient(g, x)
            return
        end

        my_gradient = no_gradient

    end

    my_objective(x) = cost(x, my_params)

    @show my_params
    my_params[:target] = target_condition(x0, my_params)

    if optimization
        (runtime, xsol, res) = run_optimization(
            case, x0,
            my_objective, my_gradient, my_options,
        )
    else
        g0 = similar(x0)
        (runtime, xsol) = run_gradient(
            case, x0, g0,
            my_objective, my_gradient,
        )
        res = nothing
    end

    sys_dir = Sys.isapple() ? "local" : "kestrel"
    mode_dir = optimization ? "optimization" : "gradient"

    fdir = joinpath(
        @__DIR__,
        "results",
        sys_dir,
        mode_dir,
        string(tarf),
    )
    mkpath(fdir)
    fbase = joinpath(
        fdir,
        "large_scale_burger_solution_$(case)_n$(Nx)_s$(seed)"
    )
    println("Saving results to: ", fbase)
    CSV.write(fbase * ".csv", Tables.table(xsol); header=false)

    dump_info(fbase * ".txt", runtime, res)

    if optimization
        # ncalls = Optim.f_calls(res)
        bp = burger_solution(xsol, my_params; save=true)
        umax = max(maximum(bp.u0), 1.0)
        umin = min(minimum(bp.u0), 0.0)
        p = plot(
            BurgersEquation.space_grid(Nx),
            BurgersEquation.expand_solution(bp.uk);
            legend=true, label="solution", ylim=(umin, umax), dpi=300,
        )
        plot!(
            p,
            BurgersEquation.space_grid(Nx),
            BurgersEquation.expand_solution(my_params[:target]),
            label="target"
        )
        plot!(
            p, BurgersEquation.space_grid(Nx), BurgersEquation.expand_solution(bp.u0),
            label="initial_condition"
        )
        png(p, fbase)
    end

    println("------------------------ Done ------------------------")

    return

end

if PROGRAM_FILE == @__FILE__
    main(ARGS)
end
