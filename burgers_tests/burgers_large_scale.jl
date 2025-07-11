import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Pkg.status()

include("burgers_common.jl")

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

function run_optimization(case, x0, f, fgrad, my_options)

    runtime = @elapsed begin

        if case == :forward

            # Optimization with FD
            println("-------- Running Optimimzation with ForwardDiff --------")
            @time res = Optim.optimize(
                f,
                x0,
                Optim.LBFGS(),
                my_options;
                autodiff=:forward, # uses ForwardDiff.jl
            )

        elseif case == :finitediff

            # Optimization with finite differences
            println("-------- Running Optimimzation with Finite Diff --------")
            @time res = Optim.optimize(
                f,
                x0,
                Optim.LBFGS(),
                my_options;
            )

        elseif case == :reverse || case == :impreverse || case == :dirreverse

            # Optimization with RD
            println("-------- Running Optimimzation with ReverseDiff --------")
            @time res = Optim.optimize(
                f,
                fgrad,
                x0,
                Optim.LBFGS(),
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
            #         # Optim.Fminbox(Optim.LBFGS()),
            #         Optim.LBFGS(),
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
                Optim.LBFGS(),
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

# if PROGRAM_FILE == @__FILE__
#     main(ARGS)
# end

main(ARGS)
