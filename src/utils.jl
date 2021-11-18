"""
    @hprofile

A macro which
- resets the default `TimerOutputs.get_defaulttimer` to zero
- execute the code block
- print the profiling details

This is useful as a coarse-grained profiling strategy to get a rough idea of
where time is spent. Note that this relies on `TimerOutputs` annotations
manually inserted in the code.
"""
macro hprofile(block)
    return quote
        TimerOutputs.enable_debug_timings(IFGF)
        reset_timer!()
        $(esc(block))
        print_timer()
    end
end

# double invsqrtQuake( double number )
#   {
#       double y = number;
#       double x2 = y * 0.5;
#       std::int64_t i = *(std::int64_t *) &y;
#       // The magic number is for doubles is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
#       i = 0x5fe6eb50c7b537a9 - (i >> 1);
#       y = *(double *) &i;
#       y = y * (1.5 - (x2 * y * y));   // 1st iteration
#       //      y  = y * ( 1.5 - ( x2 * y * y ) );   // 2nd iteration, this can be removed
#       return y;
#   }

# fast invsqrt code taken from here
# https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/nbody-julia-8.html
function invsqrt(x::Float64)
    y = @fastmath Float64(1 / sqrt(Float32(x)))
    # This is a Newton-Raphson iteration.
    return 1.5y - 0.5x * y * (y * y)
end
