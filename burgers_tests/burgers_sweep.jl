
function sbatch_script(ad::Symbol, target::Symbol, k::Int, run::Int, script::AbstractString)
    case_str = make_case_string(ad, target, k, run)
    header = """
#!/bin/bash
#SBATCH --account=diffprog
#SBATCH --time=2-0:00:00
#SBATCH --job-name=bsw_$(case_str)
#SBATCH --mail-user=jonathan.maack@nrel.gov
#SBATCH --mail-type=ALL
#SBATCH --output=bsw_$(case_str).%j.log  # %j will be replaced with the job ID
"""
    body = """
source \${HOME}/.bash_profile
module load julia/1.11
cd /projects/diffprog/jmaack/AutoDiffSVDCompression/burgers_tests

julia $script --case=$(ad) --gridsize=$(k) --seed=$(run) --target=$(target) --trace
"""
    script = header * body
    return script
end

function make_case_string(ad, target, k, run)
    return "$(ad)_$(target)_gs$(k)_$(run)"
end

function write_sbatch_script(ad::Symbol, k::Int, seed::Int, tar::Symbol, script::AbstractString)
    script_str = sbatch_script(ad, tar, k, seed, script)
    sname = "sbatch_" * make_case_string(ad, tar, k, seed) * ".sh"
    io = open(sname, "w")
    write(io, script_str)
    close(io)
    return sname
end

function main()

    RUNNING_KESTREL = Sys.islinux()
    # RUNNING_KESTREL = true

    # ad_modes = [:forward, :reverse, :finitediff, :svdforward, :svdreverse]
    ad_modes = [:forward, :reverse, :svdreverse, :finitediff]
    # ad_modes = [:svdreverse]
    grid_sizes = 4:14
    # grid_sizes = 4:6
    seeds = 1:10
    # seeds = 1:2
    targets = [:sin, :cliff]

    println("Sweeping through AD modes: ", ad_modes)
    println("With grid sizes: ", 2 .^ grid_sizes)
    println("With target: ", targets)
    println("With seeds: ", seeds)

    burger_script = joinpath(@__DIR__, "burgers_large_scale.jl")

    for tar in targets
        for k in grid_sizes
            for ad in ad_modes
                for seed in seeds
                    if (k > 11 && ad == :finitediff) || (k > 12 && ad == :forward)
                        continue
                    elseif RUNNING_KESTREL
                        sbatch_file = write_sbatch_script(ad, k, seed, tar, burger_script)
                        cmd = `sbatch $sbatch_file`
                        println(cmd)
                        run(cmd)
                    else
                        println("Running job: (", ad, ", ", k, ", ", seed, ", ", tar, ")")
                        cmd = `julia $burger_script --case=$(ad) --gridsize=$(k) --seed=$(seed) --target=$(tar) --trace`
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
