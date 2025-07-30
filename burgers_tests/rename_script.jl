
function main()

    cd("/Users/jmaack/ascr_ad/AutoDiffSVDCompression/burgers_tests/results/kestrel/optimization")
    all_files = readdir()

    for my_file in all_files
        if last(splitext(my_file)) == ".log"
            fbase = basename(my_file)
            fsplit = split(fbase, "_")
            insert!(fsplit, 2, "svdtol1e4")
            # @show fsplit
            new_file = join(fsplit, "_")
            # @show new_file
            mv(my_file, new_file)
        end
    end

    return

end

main()
