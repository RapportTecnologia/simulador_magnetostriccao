import pkgutil
import importlib
import os
import inspect

# Dynamically discover all analyzer classes in this package
_analyzers = {}
_analyzer_dir = os.path.dirname(__file__)
for _, module_name, _ in pkgutil.iter_modules([_analyzer_dir]):
    if module_name.endswith('_analyzer'):
        module = importlib.import_module(f"{__package__}.{module_name}")
        for attr in dir(module):
            if attr.endswith('Analyzer'):
                clazz = getattr(module, attr)
                if isinstance(clazz, type):
                    method = attr[:-len('Analyzer')].lower()
                    _analyzers[method] = clazz

class AnalyzerFactory:
    """
    Factory to create analyzer instances based on method name.
    """
    @staticmethod
    def create(method, sr, cutoff_freq, order=None):
        method = method.lower()
        if method not in _analyzers:
            raise ValueError(f"Unknown analysis method: {method}")
        clazz = _analyzers[method]
        # Inspect constructor signature to pass order if needed
        sig = inspect.signature(clazz.__init__)
        params = list(sig.parameters.keys())
        if 'order' in params:
            k = order if order is not None else 10
            return clazz(k, sr, cutoff_freq)
        else:
            return clazz(sr, cutoff_freq)
