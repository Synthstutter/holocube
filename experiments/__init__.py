# define __all__ for when we import * from this directory
import os
# grab all the files
__all__ = os.listdir(os.path.dirname(__file__))
# only choose ones that end in test.py, or rest.py, then drop the .py
__all__ = [exp[:-3] for exp in __all__ if exp.endswith(('rest.py', 'exp.py'))]
# sort them by rest v test, then alphabetically
__all__.sort(key=lambda f: [f[-2], f])
