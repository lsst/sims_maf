import itertools
from mafConfig import *

root.outputDir = './output2'
root.dbAddress = 'mssql+pymssql://LSST-2:L$$TUser@fatboy.npl.washington.edu:1433/LSST'
#root.dbAddress = 'postgres://calibuser:calibuser@ivy.astro.washington.edu:5432/calibDB.05.05.2010'
root.opsimNames = ['output_opsim3_61_forLynne']


constraints = ["filter = \'r\' and night < 100+49353", "filter = \'i\' and night < 100+49353"]


root.constraints = constraints

binner1 = BinnerConfig()
binner1.binner = 'HealpixBinner'
binner1.kwargs = {"nside":128}
m1 = makeMetricConfig('MeanMetric', params=['5sigma_modified'])
m2 = makeMetricConfig('RmsMetric', params=['seeing'])
binner1.metricDict = makeDict(m1,m2 )
binner1.spatialKey1 = "fieldRA"
binner1.spatialKey2 = "fieldDec"
binner1.leafsize = 50000

binner2 = BinnerConfig()
binner2.binner = 'UniBinner'
m1 = makeMetricConfig('MeanMetric', params=['5sigma_modified'])
m2 = makeMetricConfig('RmsMetric', params=['seeing'])
binner2.metricDict = makeDict(m1,m2 )

root.binners = makeDict(binner1,binner2)

