#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('Serotype_Identities.csv')


# In[3]:


z_Denmark = np.array([7, 11, 5, 21, 14, 17, 15, 3, 6, 42, 12, 22, 21, 12, 12, 11, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1])
z_Denmark = z_Denmark/sum(z_Denmark)

z_Iran = np.array([63, 58, 55, 55, 49, 41, 39, 37, 34, 33, 25, 22, 22, 20, 19, 18, 18, 17, 17, 17, 17, 17, 14, 13, 13, 10, 8, 8, 1, 1])
z_Iran = z_Iran/sum(z_Iran)

z_Brazil = np.array([19, 17, 13, 12, 10, 8, 6, 6, 5, 5, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1])
z_Brazil = z_Brazil/sum(z_Brazil)

z_Nepal = np.array([14, 5, 1, 4, 8, 25, 5, 4, 3, 2, 9, 1, 1, 2, 6, 1, 5, 2, 12, 2, 1, 1, 10, 1, 4, 19, 4, 6, 1, 9, 1, 2, 7, 3, 1, 23, 2, 4, 7, 3, 20, 2, 1, 1, 1, 2, 1, 1, 7, 2, 1, 11, 9, 7, 9, 23, 14, 7, 2, 4, 6, 2, 4, 9, 22, 2, 11, 31, 1])
z_Nepal=  z_Nepal/sum(z_Nepal)

z_Mozambique = np.array([17, 4, 1, 5, 62, 20, 1, 4, 22, 6, 26, 5, 1, 25, 40, 1, 30, 41, 33, 20, 2, 2, 4, 68, 4, 45, 10, 24, 50, 1, 19, 2, 2, 2, 40, 1, 58, 1, 20, 5, 46, 2, 4, 10, 43])
z_Mozambique = z_Mozambique/sum(z_Mozambique)


# In[5]:


Denmark = {'strain': data['Denmark'].dropna(),
          'cases':  z_Denmark}
Denmark = pd.DataFrame(Denmark).sort_values(['cases'], ascending = False)
Denmark.strain.astype('str')
        
Iran = {'strain' : data['Iran'].dropna(),
       'cases': z_Iran}
Iran = pd.DataFrame(Iran).sort_values(['cases'], ascending = False)
Iran.strain.astype('str')

Brazil = {'strain': data['Brazil'].dropna(),
         'cases': z_Brazil}
Brazil = pd.DataFrame(Brazil).sort_values(['cases'], ascending = False)
Brazil.strain.astype('str')
        
Nepal = {'strain': data['Nepal'].dropna(),
        'cases': z_Nepal}
Nepal = pd.DataFrame(Nepal).sort_values(['cases'], ascending = False)
Nepal.strain.astype('str')
        
Mozambique = {'strain': data['Mozambique'].dropna(),
             'cases': z_Mozambique}
Mozambique = pd.DataFrame(Mozambique).sort_values(['cases'], ascending = False)
Mozambique.strain.astype('str')


# In[6]:


countries = [Denmark.strain, Iran.strain, Brazil.strain, Nepal.strain, Mozambique.strain]
common_strain = list(set.intersection(*map(set, countries)))
#print(common_strain)
common_strain = ['6A', '19A', '6B', '19F', '3', '14', '4']


# In[7]:


def data_observation():
    return Iran, Denmark, Brazil, Nepal, Mozambique

def common_strain():
    return common_strain


# In[8]:


#OBSERVATION FROM 7 COUNTRIES
Den_cases = np.array(Denmark.sort_values('cases', ascending = False).cases)
Den_cases = np.append(Den_cases, np.zeros(70 - len(Den_cases)))

Ira_cases = np.array(Iran.sort_values('cases', ascending = False).cases)
Ira_cases = np.append(Ira_cases, np.zeros(70 - len(Ira_cases)))

Braz_cases = np.array(Brazil.sort_values('cases', ascending = False).cases)
Braz_cases = np.append(Braz_cases, np.zeros(70 - len(Braz_cases)))

Nep_cases = np.array(Nepal.sort_values('cases', ascending = False).cases)
Nep_cases = np.append(Nep_cases, np.zeros(70 - len(Nep_cases)))

Moz_cases = np.array(Mozambique.sort_values('cases', ascending = False).cases)
Moz_cases = np.append(Moz_cases, np.zeros(70 - len(Moz_cases)))

#compute the error between data and model
def error(case_structure):
    error = (np.linalg.norm(Den_cases - case_structure) + np.linalg.norm(Ira_cases - case_structure) + np.linalg.norm(Braz_cases - case_structure) + np.linalg.norm(Nep_cases - case_structure) + np.linalg.norm(Moz_cases - case_structure))/5
    return error

country = np.array([Ira_cases, Den_cases, Braz_cases, Nep_cases, Moz_cases])

def data_plot(axis =None):
    color = ['red', 'lime', 'fuchsia', 'aqua', 'mediumblue']
    if axis is None:
        axis = plt.gca()
    for i in range (0, len(country)):
        axis.plot(np.linspace(1, len(country[i]), len(country[i])), country[i], linewidth = 1.5, color = '{}'.format(color[i]))
        
    return axis

