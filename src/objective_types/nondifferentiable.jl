# Used for objectives and solvers where no gradient is available/exists
mutable struct NonDifferentiable{TF,TX} <: AbstractObjective
    f
    F::TF
    x_f::TX
    f_calls::Vector{Int}
end

# These could be the same if we could restrict g below not to be an AbstractArray
function NonDifferentiable(f, x::AbstractArray, F::Real = real(zero(eltype(x))); inplace = true)
    xnans = x_of_nans(x)
    NonDifferentiable{typeof(F),typeof(xnans)}(f, F, xnans, [0,])
end
function NonDifferentiable(f, x::AbstractArray, F::AbstractArray; inplace = true)
    f = !inplace && (F isa AbstractArray) ? f!_from_f(f, F, inplace) : f
    xnans = x_of_nans(x)
    NonDifferentiable{typeof(F),typeof(xnans)}(f, F, xnans, [0,])
end

# this is the g referred to above!
NonDifferentiable(f, g,        x::AbstractArray, F::Union{AbstractArray, Real} = real(zero(eltype(x)))) = NonDifferentiable(f, x, F)
NonDifferentiable(f, g, h,     x::TX, F) where TX  = NonDifferentiable(f, x, F)
NonDifferentiable(f, g, fg, h, x::TX, F) where TX  = NonDifferentiable(f, x, F)

# NonDifferentiable complex functions are reduced to real and imaginary parts
function NonDifferentiable(f, x::AbstractArray{Complex{T}}, F::Real = zero(T); inplace = true) where T
    f1(zs) = f(complex.(eachslice(zs, dims=1)...))
    x1 = similar(x, T, 2, size(x)...)
    NonDifferentiable{typeof(F), typeof(x1)}(f1, F, x1, [0,])
end
