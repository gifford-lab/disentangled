import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

comp_gene_cols = ['Dpysl4','Scube2','Prss43','Serpinf1','Fhl2','Lefty1','Lefty2','Pycr2','Cda','Tal2','Acss1','Cplx2','Lrrc4',
             'Stmn4','Tmem591','Scn1b','Tsku','Dusp14','Hoxb5os','Mreg','Pam','Matn4','S100a7a','Adamts5','Gprc5a',
                 'Lgals3','Dll3','Arl4d','Fam212a','Hoxb1','Wnt3a','Gm13715','Ptgs1','Scn9a','Vim','Saxo1','Chst7','Cited1'
                  ,'Qrfpr','Sprr2a3','Cdh2','Pcdhb2','T','Fst','Smc6','Evx1','Evx1os',
                  'Arf4','Dach1','Rarb','Ckap4','Tead2','Ccnb2','Fam64a','H2afv','Nsg2','Pbx1','Cenpa','Irs4','Mex3a','Hoxc4','Hoxc5',
                    'Ptprcap','Stk32a','Col1a2','Grifin','Igf2bp3']


group = pd.read_csv("undir_r1.concat.csv")
group = group.set_index('Unnamed: 0')

comp_gene_cols = []
comp_genes = [group.columns.get_loc(c) for c in comp_gene_cols if c in group]

profiles = np.load(sys.argv[1])
ae_profiles = profiles[0]
control_profiles = profiles[1]
target_profiles = profiles[2]

comp_gene_ccs = []
control_gene_ccs = []

for i in comp_genes:
    comp_gene_ccs.append(pearsonr(ae_profiles[i,:],target_profiles[i,:])[0])
    control_gene_ccs.append(pearsonr(control_profiles[i,:],target_profiles[i,:])[0])

plt.hist(comp_gene_ccs,color='b',alpha=0.4)
plt.hist(control_gene_ccs,color='g',alpha=0.4)
plt.show()