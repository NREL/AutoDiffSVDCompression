
function sbatch_script(
    ad::Symbol,
    target::Symbol,
    k::Int,
    run::Int,
    script::AbstractString,
    optimization::Bool
)
    case_str = make_case_string(ad, target, k, run, optimization)
    mode_arg = optimization ? "optimization" : "gradient"
    mode_dir = optimization ? "optimization" : "gradient"
    header = """
#!/bin/bash
#SBATCH --account=diffprog
#SBATCH --time=2-0:00:00
#SBATCH --job-name=bsw_$(case_str)
#SBATCH --mail-user=jonathan.maack@nrel.gov
#SBATCH --mail-type=NONE
#SBATCH --output=results/kestrel/$(mode_dir)/bsw_$(case_str).%j.log  # %j will be replaced with the job ID
"""
    body = """
source \${HOME}/.bash_profile
module load julia/1.11
cd /projects/diffprog/jmaack/AutoDiffSVDCompression/burgers_tests

export OMP_NUM_THREADS=8

julia $script --case=$(ad) --gridsize=$(k) --seed=$(run) --target=$(target) --trace --$(mode_arg)
"""
    script = header * body
    return script
end

function make_case_string(ad, target, k, run, optimization::Bool)
    mode = optimization ? "opt" : "grad"
    return "$(ad)_$(target)_$(mode)_gs$(k)_$(run)"
end

function write_sbatch_script(
    ad::Symbol,
    k::Int,
    seed::Int,
    tar::Symbol,
    script::AbstractString;
    optimization::Bool=false
)
    script_str = sbatch_script(ad, tar, k, seed, script, optimization)
    sname = "sbatch_" * make_case_string(ad, tar, k, seed, optimization) * ".sh"
    io = open(sname, "w")
    write(io, script_str)
    close(io)
    return sname
end

function main()

    RUNNING_KESTREL = Sys.islinux()
    # RUNNING_KESTREL = true

    optimization = true
    # ad_modes = [:forward, :reverse, :dirreverse, :impreverse, :svdreverse, :finitediff]
    ad_modes = [:svdreverse, :impreverse]
    # grid_sizes = 4:16
    grid_sizes = 12:12
    seeds = 1:10
    # seeds = 1:1
    targets = [:sin, :cliff, :weierstrass]
    # targets = [:sin]

    println("Sweeping through AD modes: ", ad_modes)
    println("With grid sizes: ", 2 .^ grid_sizes)
    println("With target: ", targets)
    println("With seeds: ", seeds)

    burger_script = joinpath(@__DIR__, "burgers_large_scale.jl")

    for k in grid_sizes
        for tar in targets
            for ad in ad_modes
                for seed in seeds
                    #### Optimization Excludes ####
                    if optimization &&
                       (k > 11 && ad == :finitediff
                        || k > 12 && ad == :forward
                        || k > 12 && ad == :reverse
                        || k > 12 && ad == :dirreverse
                    )
                        continue
                    end
                    #### Gradient Evaluation Excludes ####
                    if !optimization &&
                       (k > 13 && ad == :finitediff
                        || k > 14 && ad == :forward
                        || k > 13 && ad == :reverse
                        || k > 13 && ad == :dirreverse
                    )
                        continue
                    end
                    if RUNNING_KESTREL
                        sbatch_file = write_sbatch_script(ad, k, seed, tar, burger_script;
                            optimization=optimization)
                        cmd = `sbatch $sbatch_file`
                        println(cmd)
                        run(cmd)
                    else
                        println("Running job: (", ad, ", ", k, ", ", seed, ", ", tar, ")")
                        if optimization
                            cmd = `julia $burger_script --case=$(ad) --gridsize=$(k) --seed=$(seed) --target=$(tar) --trace --optimization`
                        else
                            cmd = `julia $burger_script --case=$(ad) --gridsize=$(k) --seed=$(seed) --target=$(tar) --trace --gradient`
                        end
                        println(cmd)
                        run(cmd)
                    end
                end
            end
        end
    end

    return
end

main()
