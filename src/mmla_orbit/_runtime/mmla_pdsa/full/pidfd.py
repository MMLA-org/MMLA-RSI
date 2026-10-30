"""Stable Linux process handles with an explicit, recorded compatibility backend.

Old Conda Python builds may lack pidfd bindings although the kernel supports
both syscalls. No os/signal monkey-patching and no PID-based kill fallback.
"""
import ctypes
import os
import platform
import signal
import sys


class Pidfd:
    def __init__(self, *, force_syscall=False):
        self.native_open = not force_syscall and callable(getattr(os, 'pidfd_open', None))
        self.native_send = not force_syscall and callable(getattr(signal, 'pidfd_send_signal', None))
        self.libc = None
        if not (self.native_open and self.native_send):
            if sys.platform != 'linux' or platform.machine() != 'x86_64':
                raise RuntimeError('pidfd compatibility bound only to Linux x86_64')
            self.libc = ctypes.CDLL(None, use_errno=True)
            self.libc.syscall.restype = ctypes.c_long
            self.libc.syscall.argtypes = [ctypes.c_long]

    @property
    def backend(self):
        return dict(open='native' if self.native_open else 'linux-x86_64-syscall434',
                    send='native' if self.native_send else 'linux-x86_64-syscall424')

    def _call(self, number, *args):
        ctypes.set_errno(0)
        value = self.libc.syscall(ctypes.c_long(number), *args)
        if value == -1:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return int(value)

    def open(self, pid, flags=0):
        if type(pid) is not int or not 0 < pid < 2**31 or type(flags) is not int or flags != 0:
            raise ValueError('positive PID and flags zero required')
        fd = (os.pidfd_open(pid, flags) if self.native_open else
              self._call(434, ctypes.c_int(pid), ctypes.c_uint(flags)))
        try:
            os.set_inheritable(fd, False)
            return fd
        except BaseException:
            os.close(fd)
            raise

    def send(self, fd, sig, siginfo=None, flags=0):
        if (type(fd) is not int or not 0 <= fd < 2**31 or not isinstance(sig, int)
                or not 0 <= sig < signal.NSIG or siginfo is not None
                or type(flags) is not int or flags != 0):
            raise ValueError('pidfd, bounded signal, no siginfo and zero flags required')
        if self.native_send:
            signal.pidfd_send_signal(fd, sig, None, flags)
        else:
            self._call(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(None), ctypes.c_uint(flags))

    def probe(self):
        fd = self.open(os.getpid())
        try:
            self.send(fd, 0)
            return self.backend
        finally:
            os.close(fd)
