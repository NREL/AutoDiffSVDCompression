
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include("burgers_common.jl")

import ArgParse
import CSV
import DataFrames
import LinearAlgebra
import Random

import ProgressMeter as PM
import ReverseDiff as RD

using Statistics

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

function run_gradient_error(
    Nx::Integer,
    nstart::Integer,
    ndescent::Integer,
    fgrad::Function,
    fgrad_svd::Function;
    alpha::Real=1e-1,
    seed::Integer=12345,
    progress::Bool=true,
)

    npoints = nstart * (ndescent + 1)
    umin = 0.4
    umax = 0.6

    x0 = zeros(Nx)
    random_initial_point(x0, seed, Nx, umin, umax)
    gtrue = zeros(Nx)
    gsvd = zeros(Nx)
    gerr = zeros(Nx)

    gres = zeros(npoints)
    ginf = zeros(npoints)
    gmag = zeros(npoints)
    svdmag = zeros(npoints)
    dp = zeros(npoints)

    count = 0
    nrestart = 0
    ngrad = 0

    if progress
        pm = PM.Progress(npoints)
    end

    while count < npoints

        count += 1

        # @show count
        # @show x0

        fgrad(gtrue, x0)
        fgrad_svd(gsvd, x0)

        gerr .= gsvd .- gtrue

        gres[count] = LinearAlgebra.norm(gerr, 2)
        ginf[count] = LinearAlgebra.norm(gerr, Inf)
        gmag[count] = LinearAlgebra.norm(gtrue, 2)
        svdmag[count] = Plots.LinearAlgebra.norm(gsvd, 2)
        dp[count] = LinearAlgebra.dot(gtrue, gsvd)

        if count % (ndescent + 1) == 0
            random_initial_point(x0, seed + count, Nx, umin, umax)
            nrestart += 1
            # @show ngrad, ndescent
            @assert(ngrad == ndescent)
            ngrad = 0
        else
            gradient_descent_step(x0, gtrue, alpha)
            ngrad += 1
        end

        if progress
            PM.next!(pm)
        end

    end

    # @show nrestart, nstart
    @assert(nrestart == nstart)

    if progress
        PM.finish!(pm)
    end

    df = DataFrames.DataFrame(
        :gmag => gmag,
        :svdmag => svdmag,
        :err_l2 => gres,
        :err_inf => ginf,
        :dot => dp,
    )

    return df

end

# function random_step(x0, rng, mag; scratch=similar(x0), lb=0.0, ub=2.0)

#     scratch .= mag .* randn(rng, length(x0))
#     # Keep values positive with absolute value
#     x0 .+= scratch
#     x0 .= abs.(x0 .- lb) .+ lb
#     x0 .= ub .- abs.(x0 .- ub)

#     @assert(all(x -> lb <= x <= ub, x0))

#     return

# end

# function run_random_gradient_error(
#     x_init::AbstractVector,
#     fgrad::Function,
#     fgrad_svd::Function;
#     npoints::Integer=100,
#     dx::Real=1.0,
#     rng=Random.MersenneTwister(12345),
#     progress::Bool=true,
# )

#     Nx = length(x_init)
#     x0 = copy(x_init)
#     gtrue = similar(x0)
#     gsvd = similar(x0)
#     gerr = similar(x0)

#     gres = zeros(npoints)
#     ginf = zeros(npoints)
#     gmag = zeros(npoints)
#     svdmag = zeros(npoints)
#     dp = zeros(npoints)

#     count = 0
#     # rng = Random.MersenneTwister(seed)

#     if progress
#         pm = PM.Progress(npoints)
#     end

#     while count < npoints

#         count += 1

#         fgrad(gtrue, x0)
#         fgrad_svd(gsvd, x0)

#         gerr .= gsvd .- gtrue

#         gres[count] = LinearAlgebra.norm(gerr, 2)
#         ginf[count] = LinearAlgebra.norm(gerr, Inf)
#         gmag[count] = LinearAlgebra.norm(gtrue, 2)
#         svdmag[count] = Plots.LinearAlgebra.norm(gsvd, 2)
#         dp[count] = LinearAlgebra.dot(gtrue, gsvd)

#         random_step(x0, rng, dx; scratch=gerr)

#         if progress
#             PM.next!(pm)
#         end

#     end

#     if progress
#         PM.finish!(pm)
#     end

#     df = DataFrames.DataFrame(
#         :gmag => gmag,
#         :svdmag => svdmag,
#         :err_l2 => gres,
#         :err_inf => ginf,
#         :dot => dp,
#     )

#     return df

# end

function main(ARGS)

    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        # "--gridsize"
        # help = "size of the grid to use when solving Burgers equation -- 2^gridsize"
        # arg_type = Int
        # required = true
        # "--npoints", "-n"
        # help = "number of points at which to compute gradient error"
        # arg_type = Int
        # required = true
        "--nstart", "-n"
        help = "number of random initial points at which to start"
        arg_type = Int
        required = true
        "--ndescent", "-d"
        help = "number of gradient descent steps to take from a given initial point"
        arg_type = Int
        required = true
        "--seed", "-s"
        help = "seed to send to random number generator for optimization initial condition"
        arg_type = Int
        required = true
        # "--target"
        # help = "function giving the target of function of the optimization"
        # arg_type = String
        # required = true
        "--silence"
        help = "silence progress bar"
        action = :store_true
    end

    parsed_args = ArgParse.parse_args(ARGS, s)

    grid_sizes = 4:14
    targets = [:sin, :cliff, :weierstrass]

    tf = 1.0
    cfl = 0.85

    nstart = parsed_args["nstart"]
    ndescent = parsed_args["ndescent"]
    seed = parsed_args["seed"]
    # rng = Random.MersenneTwister(seed)

    sys_dir = Sys.isapple() ? "local" : "kestrel"
    my_dir = joinpath(
        @__DIR__,
        "results",
        sys_dir,
        "error",
    )
    mkpath(my_dir)

    for k in grid_sizes

        Nx = 2^k
        (m, n) = svd_dimensions(Nx)

        for target in targets

            println("-------- Running: ($(Nx), $(target)) --------")

            fdir = joinpath(my_dir, string(target))
            mkpath(fdir)
            fbase = "grad_error_" * make_case_string(Nx, target)
            fname = joinpath(fdir, fbase * ".csv")

            if (target == :sin)
                tar_func = tf_sin
            elseif (target == :cliff)
                tar_func = tf_cliff
            elseif (target == :weierstrass)
                tar_func = tf_weierstrass
            else
                error("Unknown target function: $(tarf)")
            end

            # @show tf
            # @show Nx
            # @show tar_func
            # @show (m, n)

            # # Initial point for sin/smooth target function
            # x0 = collect(range(0.4, 0.6, Nx))
            # Initial point for cliff target function

            my_params = Dict(
                :Nx => Nx,
                :cfl => cfl,
                :tf => tf,
                :flux => :lf,
                :scale => 1e2,
                :ic => grid_control,
                :target => tar_func,
                :mode => :implicit,
            )
            my_params[:target] = target_condition(Float64[], my_params)

            svd_params = copy(my_params)
            svd_params[:mode] = :svd
            svd_params[:tol] = 1e-5
            svd_params[:matdim] = (m, n)

            my_objective(x) = cost(x, my_params)
            function my_gradient(g, x)
                RD.gradient!(g, my_objective, x)
                return
            end

            svd_objective(x) = cost(x, svd_params)
            function svd_gradient(g, x)
                RD.gradient!(g, svd_objective, x)
                return
            end

            # df = run_random_gradient_error(
            #     x0, my_gradient, svd_gradient;
            #     dx=1e-1, npoints=parsed_args["npoints"], rng=rng)

            df = run_gradient_error(
                Nx, nstart, ndescent, my_gradient, svd_gradient;
                alpha=1e-2, seed=seed,
                progress=!parsed_args["silence"]
            )

            @show maximum(df[:, :err_l2])
            @show mean(df[:, :err_l2])
            @show minimum(df[:, :err_l2])

            CSV.write(fname, df)

            println("-------- Done --------")

        end

    end

    return

end

# if PROGRAM_FILE == @__FILE__
#     main(ARGS)
# end
main(ARGS)
