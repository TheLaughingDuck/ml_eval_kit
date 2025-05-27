# ml_eval/__init__.py

'''
ml_eval_kit: A simple package for machine learning model evaluation.
'''

# Enable quick version check with `ml_eval_kit.__version__`
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("ml_eval_kit")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"