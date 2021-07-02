# Used for objectives and solvers where the gradient and Hessian is available/exists
mutable struct TwiceDifferentiable{T,TDF,TH,TX} <: AbstractObjective
    f
    df
    fdf
    dfh
    fdfh
    h
    F::T
    DF::TDF
    H::TH
    x_f::TX
    x_df::TX
    x_h::TX
    f_calls::Vector{Int}
    df_calls::Vector{Int}
    h_calls::Vector{Int}
end
# compatibility with old constructor
function TwiceDifferentiable(f, g, fg, h, x::TX, F::T = real(zero(eltype(x))), G::TG = alloc_DF(x, F), H::TH = alloc_H(x, F); inplace = true) where {T, TG, TH, TX}
    x_f, x_df, x_h = x_of_nans(x), x_of_nans(x), x_of_nans(x)

    g! = df!_from_df(g, F, inplace)
    fg! = fdf!_from_fdf(fg, F, inplace)
    h! = h!_from_h(h, F, inplace)

    TwiceDifferentiable{T,TG,TH,TX}(f, g!, fg!, nothing, nothing, h!,
                                        copy(F), copy(G), copy(H),
                                        x_f, x_df, x_h,
                                        [0,], [0,], [0,])
end

function TwiceDifferentiable(f, g, h,
                             x::AbstractVector{TX},
                             F::Real = real(zero(eltype(x))),
                             G = alloc_DF(x, F),
                             H = alloc_H(x, F); inplace = true) where {TX}
    g! = df!_from_df(g, F, inplace)
    h! = h!_from_h(h, F, inplace)

    fg! = make_fdf(x, F, f, g!)
    x_f, x_df, x_h = x_of_nans(x), x_of_nans(x), x_of_nans(x)

    return TwiceDifferentiable(f, g!, fg!, nothing, nothing, h!, F, G, H, x_f, x_df, x_h, [0,], [0,], [0,])
end



function TwiceDifferentiable(f, g,
                             x_seed::AbstractVector{T},
                             F::Real = real(zero(T)); autodiff = :finite, inplace = true) where T
    n_x = length(x_seed)

    g! = df!_from_df(g, F, inplace)
    fg! = make_fdf(x_seed, F, f, g!)

    if is_finitediff(autodiff)

        # Figure out which Val-type to use for FiniteDiff based on our
        # symbol interface.
        fdtype = finitediff_fdtype(autodiff)

        jcache = FiniteDiff.JacobianCache(x_seed, fdtype)
        function h!(storage, x)
            FiniteDiff.finite_difference_jacobian!(storage, g!, x, jcache)
            return
        end

    elseif is_forwarddiff(autodiff)
        hcfg = ForwardDiff.HessianConfig(f, copy(x_seed))
        h! = (out, x) -> ForwardDiff.hessian!(out, f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite or :forward.")
    end
    TwiceDifferentiable(f, g!, fg!, h!, x_seed, F)
end

TwiceDifferentiable(d::NonDifferentiable, x_seed::AbstractVector{T} = d.x_f, F::Real = real(zero(T)); autodiff = :finite) where {T<:Real} =
    TwiceDifferentiable(d.f, x_seed, F; autodiff = autodiff)

function TwiceDifferentiable(d::OnceDifferentiable, x_seed::AbstractVector{T} = d.x_f,
                             F::Real = real(zero(T)); autodiff = :finite) where T<:Real
    if is_finitediff(autodiff)

        # Figure out which Val-type to use for FiniteDiff based on our
        # symbol interface.
        fdtype = finitediff_fdtype(autodiff)

        jcache = FiniteDiff.JacobianCache(x_seed, fdtype)
        function h!(storage, x)
            FiniteDiff.finite_difference_jacobian!(storage, d.df, x, jcache)
            return
        end
    elseif is_forwarddiff(autodiff)
        hcfg = ForwardDiff.HessianConfig(d.f, copy(gradient(d)))
        h! = (out, x) -> ForwardDiff.hessian!(out, d.f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite or :forward.")
    end
    return TwiceDifferentiable(d.f, d.df, d.fdf, h!, x_seed, F, gradient(d))
end

function TwiceDifferentiable(f, x::AbstractArray, F::Real = real(zero(eltype(x)));
                             autodiff = :finite, inplace = true)
    if is_finitediff(autodiff)

        # Figure out which Val-type to use for FiniteDiff based on our
        # symbol interface.
        fdtype = finitediff_fdtype(autodiff)
        gcache = FiniteDiff.GradientCache(x, x, fdtype)

        function g!(storage, x)
            FiniteDiff.finite_difference_gradient!(storage, f, x, gcache)
            return
        end
        function fg!(storage, x)
            g!(storage, x)
            return f(x)
        end

        function h!(storage, x)
            FiniteDiff.finite_difference_hessian!(storage, f, x)
            return
        end
    elseif is_forwarddiff(autodiff)

        gcfg = ForwardDiff.GradientConfig(f, x)
        g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)

        fg! = (out, x) -> begin
            gr_res = DiffResults.DiffResult(zero(eltype(x)), out)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffResults.value(gr_res)
        end

        hcfg = ForwardDiff.HessianConfig(f, x)
        h! = (out, x) -> ForwardDiff.hessian!(out, f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite or :forward.")
    end
    TwiceDifferentiable(f, g!, fg!, h!, x, F)
end

function hv_product!(obj::TwiceDifferentiable, x, v)
    H = hessian!(obj, x)
    return H*v
end
