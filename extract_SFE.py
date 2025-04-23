import pandas as pd


data_num = 3000
with open('log.lammps') as f1:
    lines = f1.readlines()
SFE = []
for l in lines:
    if l.startswith('Stacking-fault energy'):
        SFE.append(l.split()[3])
data = pd.DataFrame(SFE, index=[i for i in range(1, data_num+1)], columns=['SFE'])
data.to_excel('SFE.xlsx')
