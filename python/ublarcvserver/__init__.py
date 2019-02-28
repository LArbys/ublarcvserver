import ROOT,os
if not 'UBLARCVSERVER_BASEDIR' in os.environ:
    print '$UBLARCVSERVER_BASEDIR shell env. var. not found (run configure.sh)'
    raise ImportError
if not 'LARCV_BASEDIR' in os.environ:
    print '$LARCV_BASEDIR shell env. var. not found (run configure.sh in larcv repo)'
    raise ImportError


# must load dependencies first
from larcv import larcv

ublarcvserver_dir = os.environ['UBLARCVSERVER_LIBDIR']
# We need to load in order
for l in [x for x in os.listdir(ublarcvserver_dir) if x.endswith('.so')]:
    ROOT.gSystem.Load(l)

from ROOT import ublarcvserver
