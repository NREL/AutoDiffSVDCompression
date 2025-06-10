
import Pkg
Pkg.activate(".")

using CodecZstd
using TranscodingStreams

import LinearAlgebra as LA
import SparseArrays as SA

function main()

    N = 2^8
    S = 0.1
    # mat = randn(100, 100)
    mat = Matrix(SA.sprandn(N, N, S))
    my_bytes = Vector(reinterpret(UInt8, mat[:]))
    println("Size of matrix: ", sizeof(mat))
    println("Sparsity: ", S)

    println("**** Zstd ****")
    # buf = IOBuffer()
    # stream = ZstdCompressorStream(buf)
    # write(stream, mat, TranscodingStreams.TOKEN_END)
    # compressed = take!(buf)
    # close(stream)
    codec = ZstdCompressor()
    dedec = ZstdDecompressor()
    TranscodingStreams.initialize(codec)
    TranscodingStreams.initialize(dedec)
    compressed = transcode(codec, my_bytes)

    println("Size of compressed: ", sizeof(compressed))
    println("Ratio: ", sizeof(compressed) / sizeof(mat))

    decompressed = transcode(dedec, compressed)
    matC = reshape(reinterpret(Float64, decompressed), N, N)

    TranscodingStreams.finalize(codec)
    TranscodingStreams.finalize(dedec)

    # my_bytes = read(ZstdDecompressorStream(IOBuffer(compressed)))
    # matC = reshape(reinterpret(Float64, my_bytes), N, N)

    println("Error: ", LA.norm(mat - matC, Inf))


    # println("**** Zlib ****")
    # buf = IOBuffer()
    # stream = ZlibCompressorStream(buf)
    # write(stream, mat, TranscodingStreams.TOKEN_END)
    # compressed = take!(buf)
    # close(stream)
    # println("Size of compressed: ", sizeof(compressed))
    # println("Ratio: ", sizeof(compressed) / sizeof(mat))


    # println("**** BGZF ****")
    # buf = IOBuffer()
    # stream = BGZFCompressorStream(buf)
    # write(stream, mat, TranscodingStreams.TOKEN_END)
    # compressed = take!(buf)
    # close(stream)
    # println("Size of compressed: ", sizeof(compressed))
    # println("Ratio: ", sizeof(compressed) / sizeof(mat))


    # println("**** Bzip2 ****")
    # buf = IOBuffer()
    # stream = Bzip2CompressorStream(buf)
    # write(stream, mat, TranscodingStreams.TOKEN_END)
    # compressed = take!(buf)
    # close(stream)
    # println("Size of compressed: ", sizeof(compressed))
    # println("Ratio: ", sizeof(compressed) / sizeof(mat))


    # println("**** Lz4 ****")
    # buf = IOBuffer()
    # stream = LZ4FastCompressorStream(buf)
    # write(stream, mat, TranscodingStreams.TOKEN_END)
    # compressed = take!(buf)
    # close(stream)
    # println("Size of compressed: ", sizeof(compressed))
    # println("Ratio: ", sizeof(compressed) / sizeof(mat))


    # println("**** Xz ****")
    # buf = IOBuffer()
    # stream = XzCompressorStream(buf)
    # write(stream, mat, TranscodingStreams.TOKEN_END)
    # compressed = take!(buf)
    # close(stream)
    # println("Size of compressed: ", sizeof(compressed))
    # println("Ratio: ", sizeof(compressed) / sizeof(mat))

    return

end

main()
