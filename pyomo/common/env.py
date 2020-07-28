#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import ctypes
import ctypes.wintypes
import os
import six

from six import iteritems

def _as_bytes(val):
    """Helper function to coerce a string to a bytes() object"""
    if isinstance(val, six.binary_type):
        return val
    elif val is not None:
        return val.encode('utf-8')


def _as_unicode(val):
    """Helper function to coerce a string to a unicode() object"""
    if isinstance(val, six.text_type):
        return val
    elif val is not None:
        return val.decode()


class _RestorableEnvironInterface(object):
    """Interface to track environment changes and restore state"""

    def __init__(self, dll):
        self.dll = dll
        self._original_state = {}

        # Transfer over the current os.environ
        for key, val in iteritems(os.environ):
            if val != self[key]:
                self[key] = val

        # If we can get a dictionary of the current environment (not
        # always possible), then remove any keys that are not in
        # os.environ
        origEnv = self.dll.get_env_dict()
        if origEnv is not None:
            for key in origEnv:
                if key not in os.environ:
                    del self[key]

    def restore(self):
        for key, val in iteritems(self._original_state):
            if not val:
                if self[key] is not None:
                    del self[key]
            else:
                self[key] = val
        self._original_state = {}

    def __getitem__(self, key):
        if isinstance(key, six.text_type):
            return self.dll.wgetenv(key)
        else:
            return self.dll.getenv(key)

    def __setitem__(self, key, val):
        if key not in self._original_state:
            self._original_state[key] = self[key]

        if isinstance(key, six.text_type):
            if isinstance(val, six.text_type):
                self.dll.wputenv_s(key, val)
            else:
                self.dll.wputenv_s(key, _as_unicode(val))
        elif isinstance(val, six.text_type):
            self.dll.wputenv_s(_as_unicode(key), val)
        else:
            self.dll.putenv_s(key, val)

    def __delitem__(self, key):
        if key not in self._original_state:
            self._original_state[key] = self[key]

        if isinstance(key, six.text_type):
            self.dll.wputenv_s(key, u'')
        else:
            self.dll.putenv_s(key, b'')


class _OSEnviron(object):
    """Helper class to proxy a "DLL-like" interface to os.environ"""

    _libname = 'os.environ'

    def available(self):
        return True

    def get_env_dict(self):
        return dict(os.environ)

    def getenv(self, key):
        if six.PY2:
            return _as_bytes(os.environ.get(key, None))
        else:
            # environb is not always present and depends on how the
            # interpreter was compiled.  Fall back on casting environ if
            # it is not available.
            try:
                return os.environb.get(key, None)
            except AttributeError:
                return _as_bytes(os.environ.get(_as_unicode(key),None))

    def wgetenv(self, key):
        # PY2 doesn't distinguish, and PY3's environ is nominally
        # unicode.  We will coerce the result to unicode to guarantee
        # the result type.
        return _as_unicode(os.environ.get(key, None))

    def putenv_s(self, key, val):
        # Win32 convention deletes environ entries when the string is empty
        if not val:
            if key in os.environ:
                del os.environ[key]
            return

        if six.PY2:
            os.environ[key] = val
        else:
            os.environb[key] = val

    def wputenv_s(self, key, val):
        # Win32 convention deletes environ entries when the string is empty
        if not val:
            if key in os.environ:
                del os.environ[key]
            return

        if six.PY2:
            os.environ[key] = val
        else:
            os.environ[key] = val

class _MsvcrtDLL(object):
    """Helper class to manage the interface with the MSVCRT runtime"""

    def __init__(self, name):
        self._libname = name
        if name is None:
            self._loaded = False
        else:
            self._loaded = None
        self.dll = None

    def available(self):
        if self._loaded is not None:
            return self._loaded

        try:
            self.dll = ctypes.CDLL(self._libname)
            self._loaded = True
        except OSError:
            self._loaded = False
            return self._loaded

        self.putenv_s = self.dll._putenv_s
        self.putenv_s.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.putenv_s.restype = ctypes.c_int

        self.wputenv_s = self.dll._wputenv_s
        self.wputenv_s.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        self.wputenv_s.restype = ctypes.c_int

        self.getenv = self.dll.getenv
        self.getenv.argtypes = [ctypes.c_char_p]
        self.getenv.restype = ctypes.c_char_p

        self.wgetenv = self.dll._wgetenv
        self.wgetenv.argtypes = [ctypes.c_wchar_p]
        self.wgetenv.restype = ctypes.c_wchar_p

        return self._loaded

    def get_env_dict(self):
        if not self.available():
            return None

        try:
            envp = ctypes.POINTER(ctypes.c_wchar_p).in_dll(
                self.dll, '_wenviron')
            if not envp.contents:
                envp = None
        except ValueError:
            envp = None
        if envp is None:
            try:
                envp = ctypes.POINTER(ctypes.c_char_p).in_dll(
                    self.dll, '_environ')
                if not envp.contents:
                    return None
            except ValueError:
                return None

        ans = {}
        for line in envp:
            if not line:
                break
            key, val = line.split('=', 1)
            ans[key] = val
        return ans


class _Win32DLL(object):
    """Helper class to manage the interface with the Win32 runtime"""

    def __init__(self, name):
        self._libname = name
        if name is None:
            self._loaded = False
        else:
            self._loaded = None
        self.dll = None

    def available(self):
        if self._loaded is not None:
            return self._loaded

        try:
            self.dll = ctypes.CDLL(self._libname)
            self._loaded = True
        except OSError:
            self._loaded = False
            return self._loaded

        self.putenv_s = self.dll.SetEnvironmentVariableA
        self.putenv_s.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.putenv_s.restype = ctypes.c_bool

        self.wputenv_s = self.dll.SetEnvironmentVariableW
        self.wputenv_s.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        self.wputenv_s.restype = ctypes.c_bool

        self._getenv_dll = self.dll.GetEnvironmentVariableA
        self._getenv_dll.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.wintypes.DWORD]
        self._getenv_dll.restype = ctypes.wintypes.DWORD

        self._wgetenv_dll = self.dll.GetEnvironmentVariableW
        self._wgetenv_dll.argtypes = [
            ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.wintypes.DWORD]
        self._wgetenv_dll.restype = ctypes.wintypes.DWORD

        # We (arbitrarily) choose to return the unicode environ
        self._envstr = self.dll.GetEnvironmentStringsW
        self._envstr.argtypes = []
        self._envstr.restype = ctypes.POINTER(ctypes.c_wchar)
        self._free_envstr = self.dll.FreeEnvironmentStringsW
        self._free_envstr.argtypes = [ctypes.POINTER(ctypes.c_wchar)]
        self._free_envstr.restype = ctypes.c_bool
        self._null = u'\0'

        return self._loaded

    def getenv(self, key):
        size = self._getenv_dll(key, None, 0)
        if not size:
            return None
        buf = ctypes.create_string_buffer(b'\0' * size)
        self._getenv_dll(key, buf, size)
        return buf.value

    def wgetenv(self, key):
        size = self._wgetenv_dll(key, None, 0)
        if not size:
            return None
        buf = ctypes.create_unicode_buffer(u'\0' * size)
        self._wgetenv_dll(key, buf, size)
        return buf.value

    def get_env_dict(self):
        ans = {}
        _str_buf = self._envstr()
        i = 0
        while _str_buf[i] != self._null:
            _str = ''
            while _str_buf[i] != self._null:
                _str += _str_buf[i]
                i += 1
            key, val = _str.split('=', 1)
            ans[key] = val
            i += 1 # Skip the NULL
        self._free_envstr(_str_buf)
        return ans


class CtypesEnviron(object):
    # Windows has a number of C Runtime Libraries, each of which can
    # hold its own independent copy of the system environment
    #
    # Note that this order is important: kernel32 appears to track
    # os.envion (and it is possible that it could be omitted).  It is
    # important to deal with it before the msvcrt libraries.
    DLLs = [
        _Win32DLL('kernel32'),
        _MsvcrtDLL(getattr(ctypes.util,'find_msvcrt',lambda: None)()),
        _MsvcrtDLL('api-ms-win-crt-environment-l1-1-0'),
        _MsvcrtDLL('msvcrt'),
        _MsvcrtDLL('msvcr120'),
        _MsvcrtDLL('msvcr110'),
        _MsvcrtDLL('msvcr100'),
        _MsvcrtDLL('msvcr90'),
        _MsvcrtDLL('msvcr80'),
        _MsvcrtDLL('msvcr71'),
        _MsvcrtDLL('msvcr70'),
    ]

    def __init__(self, **kwds):
        """A context manager for managing environment variables

        This class provides a simplified interface for consistently
        setting and restoring environment variables, with special
        handling to ensure consistency with the C Runtime Library
        environment on Windows platforms.

        `os.environ` reflects the current python environment variables,
        and will be passed to subprocesses.  However, it does not
        reflect the C Runtime Library (MSVCRT) environment on Windows
        platforms.  This can be problemmatic as DLLs loaded through the
        CTYPES interface will see the MSVCRT environment and not
        os.environ.  This class provides a way to manage environment
        variables and pass changes to both os.environ and the MSVCRT
        runtime.

        This class implements a context manager API, so that clients can
        temporarily change - and then subsequently restore - the
        environment.

        .. testcode::
           :hide:

           import os
           from pyomo.common.env import TemporaryEnv
           orig_env_val = os.environ.get('TEMP_ENV_VAR', None)

        .. doctest::

           >>> os.environ['TEMP_ENV_VAR'] = 'original value'
           >>> print(os.environ['TEMP_ENV_VAR'])
           original value

           >>> with CtypesEnviron(TEMP_ENV_VAR='temporary value'):
           ...    print(os.envion['TEMP_ENV_VAR'])
           temporary value

           >>> print(os.environ['TEMP_ENV_VAR'])
           original value

        .. testcode::
           :hide:

           if orig_env_val is None:
               del os.environ['TEMP_ENV_VAR']
           else:
               os.environ['TEMP_ENV_VAR'] = orig_env_val

        """
        self.interfaces = [
            _RestorableEnvironInterface(_OSEnviron()),
        ]
        self.interfaces.extend(_RestorableEnvironInterface(dll)
                               for dll in self.DLLs if dll.available())
        # Set the incoming env strings on all interfaces...
        for k, v in iteritems(kwds):
            self[k] = v

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.restore()

    def restore(self):
        """Restore the environment to the original state

        This restores all environment variables modified through this
        object to the state they were in before this instance made any
        changes.  Note that any changes made directly to os.environ
        outside this instance will not be detected / undone.

        """
        # It is possible that os.environ and the MSVCRT did not start
        # off in sync; e.g., if someone had set a value in os.environ
        # directly.  We will be especially careful and restore each
        # environ back to its original state.
        for lib in reversed(self.interfaces):
            lib.restore()

    def __getitem__(self, key):
        """Retun the current environment variable value from os.environ"""
        return os.environ[key]

    def __contains__(self, key):
        """Retun true if the key is in os.environ"""
        return key in os.environ

    def __setitem__(self, key, val):
        for lib in self.interfaces:
            lib[key] = val

    def __delitem__(self, key):
        for lib in self.interfaces:
            del lib[key]
