import numpy as np
import pandas as pd

file = pd.read_excel('C:\\Users\\msi1\\OneDrive\\Desktop\\Frequency of Purchase Analysis Data Question (2)-1.xlsx')#
#file.drop(file.columns[file.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

duplicates = pd.pivot_table(file,index = ['Outlet ID'],aggfunc='size')



f = file.groupby(['Outlet ID']).sum()

x = file['Outlet ID'].duplicated()

for i in x:
    print (len(x))


#print(x)

print(f)
