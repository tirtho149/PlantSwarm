"""
tests/conftest.py
=================
Make the repo root importable from anywhere pytest is invoked. Also
stub ``requests`` if the GPU stack isn't installed so the swarm
imports don't crash on `import requests` at module load time.
"""

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Stub requests if not installed (CI-friendly).
if "requests" not in sys.modules:
    try:
        import requests  # noqa: F401
    except ImportError:
        stub = types.ModuleType("requests")
        class _StubException(Exception): pass
        stub.RequestException = _StubException
        def _stub_call(*a, **k):
            raise _StubException("requests stub")
        stub.post = _stub_call
        stub.get  = _stub_call
        sys.modules["requests"] = stub
