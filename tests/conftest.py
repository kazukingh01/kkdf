import argparse

# Patch parse_args at module level to prevent bin modules from failing on import.
# Each bin module calls parse_args() at import time with required arguments.
_original_parse_args = argparse.ArgumentParser.parse_args

def _patched_parse_args(self, args=None, namespace=None):
    if namespace is None:
        namespace = argparse.Namespace()
    # Fill in defaults without actually parsing sys.argv
    for action in self._actions:
        if action.dest != argparse.SUPPRESS:
            if not hasattr(namespace, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(namespace, action.dest, action.default)
    try:
        return _original_parse_args(self, args=args, namespace=namespace)
    except (SystemExit, BaseException):
        return namespace

argparse.ArgumentParser.parse_args = _patched_parse_args
