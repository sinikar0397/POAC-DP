from encode import parse_unitime as parse_unitime
from POGA_DP import poga_dp_framework as poga_dp
from POAC_DP import poac_dp_framework as poac_dp
import pandas as pd
import pickle

fnames=['2019_Early_{0:02d}'.format(i) for i in range(1, 10) if (i!=4)]
fnames.append('2019_Late_01')
for fname in fnames:
    data=parse_unitime('./data/dataset_itc2019/'+fname+'.xml')
    poac_dp(data, fname+'(pheromone_ver1_4)')