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
