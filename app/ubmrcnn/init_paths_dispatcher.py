"""Add {PROJECT_ROOT}/lib. to PYTHONPATH

Usage:
import this module before import any modules under lib/
e.g
    import _init_paths
    from core.config import cfg
"""

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
one_back = osp.abspath(osp.dirname(this_dir))
net_path = osp.join(one_back, 'network')
# Add lib and net to PYTHONPATH
lib_path = osp.join(net_path, 'lib')
add_path(lib_path)
add_path(net_path)
