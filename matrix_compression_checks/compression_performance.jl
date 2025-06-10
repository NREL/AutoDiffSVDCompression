
import Pkg
Pkg.activate(".")

# using CodecBGZF
using CodecBzip2
using CodecLz4
using CodecXz
using CodecZlib
using CodecZstd
using TranscodingStreams

# using DelimitedFiles

import CSV
import DataFrames
import HeatEquation as HE
import LinearAlgebra as LA
import SparseArrays as SA

#### Compression Functions ####

# function compressor_name(compressor)
#     # if compressor == BGZFCompressorStream
#     #     return "BGZF"
#     if compressor == Bzip2CompressorStream
#         return "Bzip2"
#     elseif compressor == LZ4FastCompressorStream
#         return "LZ4Fast"
#     elseif compressor == LZ4FrameCompressorStream
#         return "LZ4Frame"
#     elseif compressor == XzCompressorStream
#         return "Xz"
#     elseif compressor == ZlibCompressorStream
#         return "Zlib"
#     elseif compressor == ZstdCompressorStream
#         return "Zstd"
#     else
#         error("Unknown compressor type")
#     end
#     return
# end

function compressor_name(compressor)
    # if compressor == BGZFCompressor
    #     return "BGZF"
    if compressor == Bzip2Compressor
        return "Bzip2"
    elseif compressor == LZ4FastCompressor
        return "LZ4Fast"
    elseif compressor == LZ4FrameCompressor
        return "LZ4Frame"
    elseif compressor == XzCompressor
        return "Xz"
    elseif compressor == ZlibCompressor
        return "Zlib"
    elseif compressor == ZstdCompressor
        return "Zstd"
    else
        error("Unknown compressor type")
    end
    return
end


#### Matrix Generation Functions ####

function as_bytes(mat::AbstractMatrix)
    return Vector(reinterpret(UInt8, mat[:]))
end

function random_matrix_as_bytes(n::Integer, sparsity::Real)
    mat = Matrix(SA.sprandn(n, n, sparsity))
    return as_bytes(mat)
end

function function_grid_matrix_as_bytes(f::Function, nx::Integer, ny::Integer)
    mat = zeros(nx, ny)
    for k in 1:nx, ell in 1:ny
        xk = HE.gridpoint(1, k, nx, Float64)
        yell = HE.gridpoint(2, k, ny, Float64)
        mat[k,ell] = f(xk, yell)
    end
    return as_bytes(mat)
end

# function get_matrix_as_bytes(args...)
#     return random_matrix_as_bytes(args...)
# end

function my_source(tk::T, xi::T, yj::T, a::T, b::T, r::T) where T
    # return T((xi - a)^2 + (yj - b)^2 <= r^2)
    val = exp(0.5*(-(xi - a)^2 - (yj - b)^2) / r^2)
end

#### Main Function ####

function main()

    compressors = [
        # BGZFCompressor,
        Bzip2Compressor,
        LZ4FastCompressor,
        LZ4FrameCompressor,
        XzCompressor,
        ZlibCompressor,
        ZstdCompressor,
    ]

    mat = randn(4,4)
    my_bytes = Vector(reinterpret(UInt8, mat[:]))
    for cmp in compressors
        # buf = IOBuffer()
        # stream = cmp(buf)
        # write(stream, mat, TranscodingStreams.TOKEN_END)
        # take!(buf)
        # close(stream)
        codec = cmp()
        TranscodingStreams.initialize(codec)
        my_cbytes = transcode(codec, my_bytes)
        TranscodingStreams.finalize(codec)
    end

    # println("******** Random Matrix ********")

    # df = DataFrames.DataFrame(
    #     :n => Int64[],
    #     :size => Int64[],
    #     :sparsity => Float64[],
    #     :compression => String[],
    #     :compressed_size => Int64[],
    #     :ratio => Float64[],
    #     :time => Float64[],
    # )

    # # for k in 1:10, S in [0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4]
    # for k in 4:12, S in 0.05:0.05:1.0

    #     N = 2^k
    #     # mat = Matrix(SA.sprandn(N, N, S))
    #     # my_bytes = Vector(reinterpret(UInt8, mat[:]))
    #     my_bytes = random_matrix_as_bytes(N, S)

    #     for cmp in compressors

    #         row_d = Dict{Symbol,Any}()
    #         row_d[:n] = N
    #         row_d[:size] = sizeof(my_bytes)
    #         row_d[:sparsity] = S
    #         row_d[:compression] = compressor_name(cmp)

    #         codec = cmp()
    #         TranscodingStreams.initialize(codec)

    #         ctime = @elapsed begin
    #             my_cbytes = transcode(codec, my_bytes)
    #             # buf = IOBuffer()
    #             # stream = cmp(buf)
    #             # write(stream, mat, TranscodingStreams.TOKEN_END)
    #             # compressed = take!(buf)
    #             # close(stream)
    #         end

    #         TranscodingStreams.finalize(codec)

    #         row_d[:compressed_size] = sizeof(my_cbytes)
    #         row_d[:ratio] = sizeof(my_cbytes) / sizeof(mat)
    #         row_d[:time] = ctime

    #         push!(df, row_d)

    #     end

    # end

    # println(df)
    # CSV.write("results_compression_random_matrix.csv", df)

    # println("******** Structured Matrix ********")

    # df = DataFrames.DataFrame(
    #     :n => Int64[],
    #     :size => Int64[],
    #     :func => String[],
    #     :p => Int64[],
    #     :q => Int64[],
    #     :compression => String[],
    #     :compressed_size => Int64[],
    #     :ratio => Float64[],
    #     :time => Float64[],
    # )

    # # fhdr = "my_f(x,y) = "

    # for p in 0:5:10, q in 0:5:10

    #     my_f(x,y) = sin(p*x)*cos(q*y)

    #     # fdef = "sin($p*x)*cos($q*y)"
    #     # eval(Meta.parse(fhdr*fdef))

    #     for k in 4:8

    #         N = 2^k
    #         # my_f(x,y) = sin(y)cos(x)
    #         my_bytes = function_grid_matrix_as_bytes(my_f, N, N)

    #         for cmp in compressors

    #             row_d = Dict{Symbol,Any}()
    #             row_d[:n] = N
    #             row_d[:size] = sizeof(my_bytes)
    #             row_d[:func] = "sin($(p)x)cos($(q)y)"
    #             row_d[:p] = p
    #             row_d[:q] = q
    #             row_d[:compression] = compressor_name(cmp)

    #             codec = cmp()
    #             TranscodingStreams.initialize(codec)

    #             ctime = @elapsed begin
    #                 my_cbytes = transcode(codec, my_bytes)
    #             end

    #             TranscodingStreams.finalize(codec)

    #             row_d[:compressed_size] = sizeof(my_cbytes)
    #             row_d[:ratio] = sizeof(my_cbytes) / sizeof(my_bytes)
    #             row_d[:time] = ctime

    #             push!(df, row_d)

    #         end

    #     end

    # end

    # println(df)
    # CSV.write("results_compression_structured_matrix.csv", df)

    println("******** Heat Solution Matrix ********")

    df = DataFrames.DataFrame(
        :n => Int64[],
        :size => Int64[],
        :Tf => Float64[],
        :udiff_norm => Float64[],
        :compression => String[],
        :compressed_size => Int64[],
        :ratio => Float64[],
        :time => Float64[],
    )

    kappa = 1.0
    dt = 5e-3
    dT = 0.5
    f(t,x,y) = my_source(t, x, y, -0.0, 0.0, 0.5)

    for k in 4:10

        println("    #### k = $k ####")

        N = 2^k
        u0 = zeros(N, N)
        ud = copy(u0)
        @time he_prob = HE.heat_setup_cpu(u0, kappa, f, dT, dt, N, N, -1, :csc)

        for tf in dT:dT:10.0

            @time HE.heat_loop(he_prob, nothing; progress=false)
            u1 = he_prob.uk
            ud .= u1 .- u0
            udiff = LA.norm(ud, 2) / (N*N)
            u0 .= u1

            @show udiff

            my_bytes = as_bytes(u1)

            for cmp in compressors

                row_d = Dict{Symbol,Any}()
                row_d[:n] = N
                row_d[:size] = sizeof(my_bytes)
                row_d[:Tf] = tf
                row_d[:udiff_norm] = udiff
                row_d[:compression] = compressor_name(cmp)

                codec = cmp()
                TranscodingStreams.initialize(codec)

                ctime = @elapsed begin
                    my_cbytes = transcode(codec, my_bytes)
                end

                TranscodingStreams.finalize(codec)

                row_d[:compressed_size] = sizeof(my_cbytes)
                row_d[:ratio] = sizeof(my_cbytes) / sizeof(my_bytes)
                row_d[:time] = ctime

                push!(df, row_d)

            end

        end

    end

    println(df)
    CSV.write("results_compression_heat_matrix.csv", df)


    return

end

main()
