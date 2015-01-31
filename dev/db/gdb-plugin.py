print("---")
print("--- Loading CUDA-EDU Plugin")
print("---")

import sys

################################################################################
###
### Misc Util
###
def source_lines():
    main = gdb.lookup_global_symbol("main")
    symtab = main.symtab
    return sorted(symtab.linetable().source_lines())

def source_line():
    return gdb.selected_frame().find_sal().line

def function_symbol():
    return gdb.newest_frame().function()

def current_threadIdx():
    return gdb.parse_and_eval("edu::cuda::current_cuda_thread()->threadIdx")

def current_blockIdx():
    return gdb.parse_and_eval("edu::cuda::current_cuda_thread()->blockIdx")

def this_threadIdx():
    return gdb.parse_and_eval("this->threadIdx")

def in_device():
    return str(gdb.parse_and_eval("edu::mem::curr_space")) == "edu::mem::MemorySpace_Device"


################################################################################
###
### Breakpoints
###

#
# YieldBreakpoint
#
# Detects when we're switching away from a thread.
#
class YieldBreakpoint (gdb.Breakpoint):
    SPEC="edu::cuda::cuda_thread_t::yield"

    def __init__(self, next_cmd):
        super(YieldBreakpoint, self).__init__(YieldBreakpoint.SPEC, internal=True)
        self.silent = True

        self.next_cmd = next_cmd

    def stop(self):
        self.next_cmd.create_resume_breakpoint()
        return False

#
# ResumeThreadBreakpoint
#
# Detects when particular thread resumes, at which point "next breakpoints"
# are re-enabled.
#
class ResumeThreadBreakpoint (gdb.Breakpoint):
    SPEC="edu::cuda::cuda_thread_t::resume"

    def __init__(self, next_cmd, tidx):
        super(ResumeThreadBreakpoint, self).__init__(ResumeThreadBreakpoint.SPEC, internal=True)
        self.silent = True

        self.next_cmd = next_cmd
        self.tidx = tidx
        self.n = 0
        self.first_resume = None
        self.last_resume = None

    def stop(self):
        tidx = this_threadIdx()
        if tidx == self.tidx:
            if self.n > 0:
                print("[Executed threads %s - %s]" % (self.first_resume, self.last_resume))
            self.next_cmd.resume()
        else:
            if self.n == 0:
                self.first_resume = tidx
            self.last_resume = tidx
            self.n += 1
        return False

#
# ResumeAnyBreakpoint
#
# Detects when any thread resumes.
#
class ResumeAnyBreakpoint (gdb.Breakpoint):
    SPEC="edu::cuda::cuda_thread_t::resume"

    def __init__(self, cmd):
        super(ResumeAnyBreakpoint, self).__init__(ResumeAnyBreakpoint.SPEC,
                                                  internal=True)
        self.silent = True

        self.cmd = cmd

    def stop(self):
        self.cmd.resume()
        return False

#
# NextBreakpoint
#
# Placed at every line of main module when we're stepping a line.
#
class NextBreakpoint (gdb.Breakpoint):
    def __init__(self, source_line, next_cmd_func_sym):
        super(NextBreakpoint, self).__init__(str(source_line), internal=True)
        self.silent = True

        self.next_cmd_func_sym = next_cmd_func_sym

    def stop(self):
        func_sym = function_symbol()
        if self.next_cmd_func_sym.linkage_name == func_sym.linkage_name and self.enabled:
            self.enabled = False
            line = source_line()
            gdb.execute("list %d,%d" % (line,line))
            return True
        else:
            return False

################################################################################
###
### Commands
###

#
# NextCommand
#
class NextCommand (gdb.Command):
    def __init__ (self, any_thread):
        super (NextCommand, self).__init__ ("n" if any_thread else "tnext", gdb.COMMAND_USER)
        self.resume_any_thread = any_thread

    def invoke (self, arg, from_tty):
        if not in_device():
            gdb.execute("next")
        else:
            self.create_yield_and_next_breakpoints()
            self.resume_breakpoint = None
            gdb.execute("cont")

    def create_yield_and_next_breakpoints(self):
        func_sym = function_symbol()
        self.next_breakpoints = []
        for sl in source_lines():
            self.next_breakpoints.append(NextBreakpoint(sl, func_sym))
        self.yield_breakpoint = YieldBreakpoint(self)
        gdb.events.stop.connect(NextCommand.AllBreakpointsDisposer(self))

    def enable_next_breakpoints(self):
        for brk in self.next_breakpoints:
            brk.enabled = True

    def disable_next_breakpoints(self):
        for brk in self.next_breakpoints:
            brk.enabled = False

    def create_resume_breakpoint(self):
        if self.resume_any_thread:
            self.resume_breakpoint = ResumeAnyBreakpoint(self)
        else:
            # Should be able to use this_threadIdx()?
            self.resume_breakpoint = ResumeThreadBreakpoint(self, this_threadIdx())

        self.disable_next_breakpoints()
        self.delete_yield_breakpoint()

    def resume(self):
        self.enable_next_breakpoints()
        self.delete_resume_breakpoint()

    def delete_yield_breakpoint(self):
        NextCommand.SingleBreakpointDisposer(self, "yield")

    def delete_resume_breakpoint(self):
        NextCommand.SingleBreakpointDisposer(self, "resume")

    class SingleBreakpointDisposer:
        def __init__(self, next_cmd, brk_type):
            if brk_type == "yield":
                self.brk = next_cmd.yield_breakpoint
                next_cmd.yield_breakpoint = None
            elif brk_type == "resume":
                self.brk = next_cmd.resume_breakpoint
                next_cmd.resume_breakpoint = None
            self.brk.enabled = False
            gdb.post_event(self)

        def __call__(self):
            self.brk.delete()

    class AllBreakpointsDisposer:
        def __init__(self, next_cmd):
            self.next_cmd = next_cmd
            
        def __call__(self, event):
            for brk in self.next_cmd.next_breakpoints:
                brk.delete()
            del self.next_cmd.next_breakpoints
            if self.next_cmd.yield_breakpoint:
                self.next_cmd.yield_breakpoint.delete()
            if self.next_cmd.resume_breakpoint:
                self.next_cmd.resume_breakpoint.delete()
            gdb.events.stop.disconnect(self)

################################################################################
###
### Prompt
###
class Prompt:
    def __init__(self):
        self.tidx = None
        self.bidx = None

    def __call__(self, current_prompt):
        self.update_idx()
        self.show_status()
        return "(cuda-edu) "

    def update_idx(self):
        self.prev_tidx = self.tidx
        self.prev_bidx = self.bidx
        if not len(gdb.selected_inferior().threads()):
            self.tidx = None
            self.bidx = None
            return
        if in_device():
            self.tidx = current_threadIdx()
            self.bidx = current_blockIdx()

    def show_status(self):
        if self.tidx and ((self.tidx != self.prev_tidx) or (self.bidx != self.prev_bidx)):
            print("[threadIdx=%s, blockIdx=%s]" % (self.tidx, self.bidx))


################################################################################
###
### Plugin Init
###
NextCommand(True)
NextCommand(False)
gdb.prompt_hook = Prompt()
