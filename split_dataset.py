import os
from random import shuffle

datadir = 'data/jazz_xlab/'
if os.path.exists(datadir+'.DS_Store'):
    os.remove(datadir+'.DS_Store')
files = os.listdir(datadir)
shuffle(files)

ratiotrain = 0.8
ratiotest = 0.1
Nfiles = len(files)
Ntrain = int(Nfiles*ratiotrain)
Ntest = int(Nfiles*ratiotest)
train_files = files[0:Ntrain]
test_files = files[Ntrain:Ntrain+Ntest]
valid_files = files[Ntrain+Ntest:Nfiles]

for file in train_files:
    os.replace(datadir+file, 'data/train/'+file)
for file in test_files:
    os.replace(datadir+file, 'data/test/'+file)
for file in valid_files:
    os.replace(datadir+file, 'data/validation/'+file)
