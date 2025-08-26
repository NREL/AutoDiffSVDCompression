
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include("burgers_common.jl")

import ArgParse
import CSV
import DataFrames
import Distributed
import LinearAlgebra
import Random

import ProgressMeter as PM
import ReverseDiff as RD

using Statistics

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
    umin = 0.45
    umax = 0.55

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
        svdmag[count] = LinearAlgebra.norm(gsvd, 2)
        dp[count] = LinearAlgebra.dot(gtrue, gsvd)

        if count % (ndescent + 1) == 0
            # @show seed + count
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

function run_gradient_error_dist(
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

    gres = zeros(npoints)
    ginf = zeros(npoints)
    gmag = zeros(npoints)
    svdmag = zeros(npoints)
    dp = zeros(npoints)

    pm = PM.Progress(npoints; enabled=progress)

    # Distributed.@distributed for job in 1:nstart
    #     gradient_error_inner_loop(pm,
    #         seed, ndescent, Nx, fgrad, fgrad_svd, alpha,
    #         job)
    # end

    # println("Launching jobs.")

    # Send jobs to workers
    size = max(Distributed.nprocs(), 10)
    result_channel = Distributed.RemoteChannel(
        () -> Channel{MyResult}(size)
    )

    rd = Dict{Int,Distributed.Future}()
    for job in 1:nstart
        rd[job] = Distributed.@spawnat(
            :any,
            gradient_error_inner_loop(result_channel,
                seed, ndescent, Nx,
                fgrad, fgrad_svd, alpha, job)
        )
    end

    # println("Launched ", length(rd), " jobs.")

    # println("Gathering results.")

    running = true
    while running

        # println("Checking for results")

        while isready(result_channel)
            result = take!(result_channel)
            # MyResult(my_idx, gres, ginf, gmag, svdmag, dp)
            idx = result.idx
            # println("Got results for index: ", idx)
            gres[idx] = result.gres
            ginf[idx] = result.ginf
            gmag[idx] = result.gmag
            svdmag[idx] = result.svdmag
            dp[idx] = result.dp

            PM.next!(pm)
        end

        # println("Checking workers")

        for (pid, f) in pairs(rd)
            if isready(f)
                # Do not care about return value unless it is an error
                fetch(f)
                # println("Job $pid completed.")
                delete!(rd, pid)
                break
            end
        end

        sleep(1)

        if length(rd) == 0
            running = false
        end

    end

    PM.finish!(pm)

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
        "--nworkers", "-w"
        help = "number of worker processes to use"
        arg_type = Int
        default = 0
        required = false
        "--seed", "-s"
        help = "seed to send to random number generator for optimization initial condition"
        arg_type = Int
        required = true
        # "--target"
        # help = "function giving the target of function of the optimization"
        # arg_type = String
        # required = true
        "--no-save"
        help = "run computation without saving results"
        action = :store_true
        "--silence"
        help = "silence progress bar"
        action = :store_true
    end

    parsed_args = ArgParse.parse_args(ARGS, s)

    # grid_sizes = 4:4
    grid_sizes = 4:10
    # grid_sizes = 14:14
    # targets = [:sin, :cliff, :weierstrass]
    targets = [:weierstrass]

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

    if (parsed_args["nworkers"] > 1)
        Distributed.addprocs(parsed_args["nworkers"])
        Distributed.@everywhere include("burgers_common.jl")
    end

    for k in grid_sizes

        Nx = 2^k
        (m, n) = svd_dimensions(Nx)

        for target in targets

            println("-------- Running: ($(Nx), $(target)) --------")

            fdir = joinpath(my_dir, string(target))
            mkpath(fdir)
            fbase = "grad_error_1em5_" * make_case_string(Nx, target)
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
                :a => 1.0,
                :b => 10.0,
            )
            my_params[:target] = target_condition(Float64[], my_params)

            svd_params = copy(my_params)
            svd_params[:mode] = :svd
            svd_params[:nsv] = 1
            # svd_params[:tol] = 1e-5
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

            # df = run_gradient_error_dist(
            #     Nx, nstart, ndescent, my_gradient, svd_gradient;
            #     alpha=1e-2, seed=seed,
            #     progress=!parsed_args["silence"]
            # )

            @show maximum(df[:, :err_l2])
            @show mean(df[:, :err_l2])
            @show minimum(df[:, :err_l2])

            if !parsed_args["no-save"]
                CSV.write(fname, df)
            end

            println("-------- Done --------")

        end

    end

    return

end

function test_mt()

    seed = 1
    nstart = 2
    ndescent = 5

    tf = 1.0
    cfl = 0.85
    Nx = 2^8
    (m, n) = svd_dimensions(Nx)

    my_params = Dict(
        :Nx => Nx,
        :cfl => cfl,
        :tf => tf,
        :flux => :lf,
        :scale => 1e2,
        :ic => grid_control,
        :target => tf_sin,
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

    serial_df = run_gradient_error(
        Nx, nstart, ndescent, my_gradient, svd_gradient;
        alpha=1e-2, seed=seed,
        progress=true,
    )

    Distributed.addprocs(2)
    Distributed.@everywhere(include("burgers_common.jl"))

    multi_df = run_gradient_error_dist(
        Nx, nstart, ndescent, my_gradient, svd_gradient;
        alpha=1e-2, seed=seed,
        progress=true
    )

    return (serial_df, multi_df)

end

# if PROGRAM_FILE == @__FILE__
#     main(ARGS)
# end
main(ARGS)
# @show test_mt()
