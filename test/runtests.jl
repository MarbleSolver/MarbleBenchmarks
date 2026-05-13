using Test

@testset "problem files define matching function" begin
    for f in readdir(joinpath(@__DIR__, "..", "problems"), join=true)
        endswith(f, ".jl") || continue
        fname = splitext(basename(f))[1]
        src = read(f, String)
        @testset "$fname" begin
            @test occursin(Regex("function\\s+$(fname)\\b"), src)
        end
    end
end
