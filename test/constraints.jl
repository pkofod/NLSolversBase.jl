@testset "Constraints" begin
    @testset "ConstraintBounds" begin
        cb = ConstraintBounds([], [], [0.0], [0.0])
        @test NLSolversBase.nconstraints(cb) == 1
        @test NLSolversBase.nconstraints_x(cb) == 0
        @test cb.valc == [0.0]
        @test eltype(cb) == Float64

        cb = ConstraintBounds([0], [0], [], [])
        @test NLSolversBase.nconstraints(cb) == 0
        @test NLSolversBase.nconstraints_x(cb) == 1
        @test cb.valx == [0]
        @test eltype(cb) == Int
        io = IOBuffer()
        show(io, cb)
        s = String(take!(io))
        @test s == "ConstraintBounds:\n  Variables:\n    x[1]=0\n  Linear/nonlinear constraints:"
        cb = ConstraintBounds([], [3.0], [0.0], [])
        @test NLSolversBase.nconstraints(cb) == 1
        @test NLSolversBase.nconstraints_x(cb) == 1
        @test cb.bx[1] == 3.0
        @test cb.σx[1] == -1
        @test cb.bc[1] == 0.0
        @test cb.σc[1] == 1
        @test eltype(cb) == Float64

        cb = ConstraintBounds([],[],[],[])
        @test eltype(cb) == Union{}
        @test eltype(convert(ConstraintBounds{Int}, cb)) == Int

        cb =  ConstraintBounds([1,2], [3,4.0], [], [10.,20.,30.])
        io = IOBuffer()
        show(io, cb)
        s = String(take!(io))
        @test s == "ConstraintBounds:\n  Variables:\n    x[1]≥1.0, x[1]≤3.0, x[2]≥2.0, x[2]≤4.0\n  Linear/nonlinear constraints:\n    c_1≤10.0, c_2≤20.0, c_3≤30.0"

    end

    @testset "Once differentiable constraints" begin
        lx, ux = (1.0,2.0)
        odc = OnceDifferentiableConstraints([lx], [ux])
        @test odc.bounds.bx == [lx, ux]
        @test isempty(odc.bounds.bc)

        prob = MVP.ConstrainedProblems.examples["HS9"]
        cbd = prob.constraintdata
        cb = ConstraintBounds(cbd.lx, cbd.ux, cbd.lc, cbd.uc)

        odc = OnceDifferentiableConstraints(cb)
        @test odc.bounds == cb

        odc = OnceDifferentiableConstraints(cbd.c!, cbd.jacobian!,
                                            cbd.lx, cbd.ux, cbd.lc, cbd.uc)
        @test isempty(odc.bounds.bx)
        @test isempty(odc.bounds.bc)
        @test odc.bounds.valc == [0.0]

        #TODO: add tests calling c! and jacobian!

        @testset "autodiff" begin
            # This throws because cbd.lc is empty (when using problem "HS9")
            @test_throws ArgumentError OnceDifferentiableConstraints(cbd.c!, cbd.lx, cbd.ux,
                                                                     cbd.lc, cbd.uc)

            lx, ux, lc, uc = cbd.lx, cbd.ux, cbd.lc, cbd.uc
            nx = length(prob.initial_x)
            nc = length(cbd.lc)
            if isempty(lx)
                lx = fill(-Inf, nx)
                ux = fill(Inf, nx)
            end

            for autodiff in (:finite, :forward)
                odca = OnceDifferentiableConstraints(cbd.c!, lx, ux,
                                                     lc, uc, autodiff)

                T = eltype(odca.bounds)
                carr = zeros(T, nc)
                carra = similar(carr)
                odc.c!(carr, prob.initial_x)
                odca.c!(carra, prob.initial_x)

                @test carr == carra

                Jarr  = zeros(T, nc, nx)
                Jarra = similar(Jarr)
                odc.jacobian!(Jarr, prob.initial_x)
                odca.jacobian!(Jarra, prob.initial_x)

                @test isapprox(Jarr, Jarra, atol=1e-10)
            end
        end
    end

    @testset "Twice differentiable constraints" begin
    
        odc = TwiceDifferentiableConstraints([lx], [ux])
        @test odc.bounds.bx == [lx, ux]
        @test isempty(odc.bounds.bc)

        prob = MVP.ConstrainedProblems.examples["HS9"]
        cbd = prob.constraintdata
        cb = ConstraintBounds(cbd.lx, cbd.ux, cbd.lc, cbd.uc)

        odc = TwiceDifferentiableConstraints(cb)
        @test odc.bounds == cb

        odc = TwiceDifferentiableConstraints(cbd.c!, cbd.jacobian!, cbd.h!,
                                            cbd.lx, cbd.ux, cbd.lc, cbd.uc)
        @test isempty(odc.bounds.bx)
        @test isempty(odc.bounds.bc)
        @test odc.bounds.valc == [0.0]

        @testset "autodiff" begin

            cb = ConstraintBounds(cbd.lx, cbd.ux, cbd.lc, cbd.uc)
            nc = length(cbd.lc)
            nx = length(prob.initial_x)
            lx = fill(-Inf, nx)
            ux = fill(Inf, nx)
            odc = TwiceDifferentiableConstraints(cbd.c!, cbd.jacobian!, cbd.h!,lx, ux, cbd.lc, cbd.uc)

            T = eltype(odca.bounds)
            nx = length()
            jac_result = zeros(T, nc,nx)
            jac_result_autodiff = zeros(T, nc,nx)

            hess_result = zeros(T, nx,nx)
            hess_result_autodiff = zeros(T, nx,nx)
            λ = rand(T,nc)

            for autodiff in (:finite, :forward) #testing double differentiation
                odca2 = TwiceDifferentiableConstraints(cbd.c!,,lx, ux, cbd.lc, cbd.uc)
                
                odca2.jacobian!(jac_result_autodiff,prob.initial_x) 
                odc.jacobian!(jac_result,prob.initial_x)
                
                odca2.h!(hess_result_autodiff,λ,prob.initial_x) 
                odc.h!(hess_result,λ,prob.initial_x)

                @test isapprox(jac_result, jac_result_autodiff, atol=1e-10) 
                @test isapprox(hess_result, hess_result_autodiff, atol=1e-10) 
            end

            for autodiff in (:finite, :forward) #testing autodiff hessian from constraint jacobian
                odca2 = TwiceDifferentiableConstraints(cbd.c!, cbd.jacobian!,lx, ux, cbd.lc, cbd.uc)
                odca2.h!(hess_result_autodiff,λ,prob.initial_x) 
                odc.h!(hess_result,λ,prob.initial_x)
                @test isapprox(hess_result, hess_result_autodiff, atol=1e-10) 
            end
        end
    end
end

