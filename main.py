from encode import parse_unitime as parse_unitime
from POGA_DP import poga_dp_framework as poga_dp

fname='2019_Early_01'
datapath='./data/dataset_itc2019/'+fname+'.xml'
data=parse_unitime(datapath)
schedule, room = poga_dp(data, fname)
print(schedule)


