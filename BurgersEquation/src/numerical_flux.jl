
function lax_friedrichs(f::Function, fprime::Function, u::Real, v::Real, ::Real)
    fu = f(u)
    fv = f(v)
    # Below assumes f is convex so that fprime is an increasing function
    alpha = max(abs(fprime(u)), abs(fprime(v)))
    return 0.5 * (fu + fv - alpha*(v - u))
end

function lax_wendroff(f::Function, fprime::Function, u::Real, v::Real, ratio::Real)
    # fu = u^2
    # fv = v^2
    # fp = u + v
    fu = f(u)
    fv = f(v)
    fp = fprime(0.5*(u + v))
    return 0.5 * (fu + fv - ratio*fp*(fv - fu))
end
