
function main()

    # ad_modes = [:forward, :reverse, :finitediff, :svdforward, :svdreverse]
    ad_modes = [:forward, :reverse, :svdreverse, :finitediff]
    # grid_sizes = 4:14
    grid_sizes = 4:14

    println("Sweeping through AD modes: ", ad_modes)
    println("With grid sizes: ", 2 .^ grid_sizes)

    burger_script = joinpath(@__DIR__, "burgers_large_scale.jl")

    for k in grid_sizes
        for ad in ad_modes
            cmd = `julia $burger_script --case=$(ad) --gridsize=$(k) --trace`
            println(cmd)
            run(cmd)
        end
    end

    return
end

main()
