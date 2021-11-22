using LoopVectorization

function _helmholtz3d_sl_fast!(C,X,Y,σ,k)
    m,n = size(X,2), size(Y,2)
    C_T = reinterpret(Float64, C)
    C_r = @views C_T[1:2:end,:]
    C_i = @views C_T[2:2:end,:]
    σ_T = reinterpret(Float64, σ)
    σ_r = @views σ_T[1:2:end,:]
    σ_i = @views σ_T[2:2:end,:]
    @turbo for j in 1:n
        for i in 1:m
            d2 = (X[1,i] - Y[1,j])^2
            d2 += (X[2,i] - Y[2,j])^2
            d2 += (X[3,i] - Y[3,j])^2
            d  = sqrt(d2)
            s, c = sincos(k * d)
            zr = inv(4π*d) * c
            zi = inv(4π*d) * s
            C_r[i] += zr*σ_r[j] - zi*σ_i[j]
            C_i[i] += zi*σ_r[j] + zr*σ_i[j]
        end
    end
    return C
end

function _laplace3d_sl_fast!(C,X,Y,σ)
    m,n = size(X,2), size(Y,2)
    @turbo for j in 1:n
        for i in 1:m
            d2 = (X[1,i] - Y[1,j])^2
            d2 += (X[2,i] - Y[2,j])^2
            d2 += (X[3,i] - Y[3,j])^2
            # fast invsqrt code taken from here
            # https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/nbody-julia-8.html
            invd    = @fastmath Float64(1 / sqrt(Float32(d2)))
            invd = 1.5invd - 0.5d2 * invd * (invd * invd)
            C[i] += inv(4π)*invd*σ[j]
            # C[i] += inv(4π*sqrt(d2))*σ[j] # significalty slower
        end
    end
    return C
end
