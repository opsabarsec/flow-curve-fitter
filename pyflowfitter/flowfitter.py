# -*- coding: utf-8 -*-
"""
Rheology Flow Curve Models.
"""

# Author: Marco Berta <marco.berta@alumni.manchester.ac.uk>
# https://fr.linkedin.com/in/marco-berta/

# Version 1.0
# License: BSD 3 clause

import numpy as np
import pandas as pd

# Import curve fitting package from scipy
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error


# In[2]:

print('verify that this script comes together the file input_data.csv')
print('the csv spreadsheed contains a column for shear stress and one for strain rate')
print("update it with your own experimental values if you haven't done it yet")
print('')
# import from csv containing stress and strain rate and delete eventual empty cells
df_ares = pd.read_csv('input_data.csv').dropna() 
#define viscosity. We calculate it rather than importing it from the instrument software
df_ares['Viscosity'] =  df_ares['Shear stress']/df_ares['Shear rate'] 
df_ares.head() #visualize input table


# In[3]:


# extract numpy arrays from the dataframe columns
x1 = df_ares['Shear rate'].values
y_stress = df_ares['Shear stress'].values
y_visc = df_ares['Viscosity'].values


# ## 2. Modelling : equation coefficients calculation

# equations from:
# https://blog.rheosense.com/modeling-non-newtonian-fluids

# ### 2.1 Models for non-newtonian fluids

# In[4]:


# Function to calculate the Ostwald de Waele (power law) model 
def power(x,  K, n):
    return K*x**(n-1)

# Function to calculate the Carreau model
# from https://en.wikipedia.org/wiki/Carreau_fluid
def Carreau(x ,eta0, eta_inf, L, n):
    return eta_inf + (eta0 - eta_inf)*(1 + (L*x)**2)**((n-1)/2)

# Function to calculate the Yasuda model
def Yasuda(x ,eta0, eta_inf, L, n,a):
    return (eta0-eta_inf) * ((1 + (L*x)**a)**((n-1)/a))

# Function to calculate the Cross model
def Cross(x ,eta0, eta_inf, k, n):
    return eta_inf + (eta0-eta_inf)/(1 +(k*x)**n)





def non_newtonian_fits(x,y):
    non_newtonian_models = [power, Carreau, Yasuda, Cross]
    model_param_list = []
    for model in non_newtonian_models:
        param, param_cov = curve_fit(model, x,y , maxfev=5000)
        model_param_list.append(np.round(param,2))
    return model_param_list

NonNewtonian_param_list = non_newtonian_fits(x1, y_visc)

NonNewtonian_param_list 





# ### 2.2 Models that assume that the fluid has a yield stress

# In[7]:


# Function to calculate the Bingham model
def Bingham(x ,y0, pv):
    return y0 + pv*x
# https://glossary.oilfield.slb.com/en/terms/b/bingham_plastic_model
 
# Function to calculate the HB model
def Herschel_Bulkley(x ,y0, K, n):
    return (y0 + K*x**n)

# Function to calculate the HB model
def Casson(x ,yc, eta):
    return ((np.sqrt(yc) + np.sqrt(eta*x))**2)/1


# In[8]:


def yield_stress_fits(x,y):
    yield_stress_models = [Bingham, Herschel_Bulkley, Casson]
    model_param_list = []
    for model in yield_stress_models:
        param, param_cov = curve_fit(model, x,y , maxfev=5000)
        model_param_list.append(np.round(param,2))
    return model_param_list

yield_stress_param_list = yield_stress_fits(x1, y_stress)

 


# ### 3. Modelling : viscosity values from each fit

# In[9]:


# now let's calculate the viscosity values from the fit
eta_pow = NonNewtonian_param_list[0][0] * (x1**(NonNewtonian_param_list[0][1]-1))

# Carreau parameters

eta0 = NonNewtonian_param_list[1][0]
eta_inf = NonNewtonian_param_list[1][1]
L = NonNewtonian_param_list[1][2]
n = NonNewtonian_param_list[1][3]


eta_car = eta_inf + (eta0 - eta_inf)*(1 + (L*x1)**2)**((n-1)/2)

# Yasuda parameters
eta0 = NonNewtonian_param_list[2][0]
eta_inf = NonNewtonian_param_list[2][1]
L = NonNewtonian_param_list[2][2]
n = NonNewtonian_param_list[2][3]
a = NonNewtonian_param_list[2][4]

eta_yas = (eta0-eta_inf) * ((1 + (L*x1)**a)**((n-1)/a))
#cross parameters
c_eta0 = NonNewtonian_param_list[3][0]
c_eta_inf = NonNewtonian_param_list[3][1]
c_k = NonNewtonian_param_list[3][2]
c_n = NonNewtonian_param_list[3][3]

eta_cross = c_eta_inf + (c_eta0-c_eta_inf)/(1 +(c_k*x1)**c_n)


# In[10]:


# now let's calculate the stress and viscosity values from the fit

y_bin = yield_stress_param_list[0][0] + yield_stress_param_list[0][1]*x1
eta_bin = y_bin/x1

y_hb =   yield_stress_param_list[1][0] + yield_stress_param_list[1][1]*x1**yield_stress_param_list[1][2]
eta_hb = y_hb/x1

y_cas = (np.sqrt(yield_stress_param_list[2][0]) + np.sqrt(yield_stress_param_list[2][1]*x1))**2
eta_cas = y_cas/x1


# ### 4. Models scores

# In[11]:


def compare_models(eta_list):    
    MLA_names = []
    ExplVar_scores = [] # explained variance list
    MAE_scores = [] # mean average error list
    RMSE_scores = [] # root mean square error list
    R2_scores = [] # regression coefficients list
    MLA_names = ['Ostwald – de Waele power law', 'Carreau', 'Carreau-Yasuda', 'Cross', 'Bingham',"Herschel-Bulkley", 'Casson']
       
    for y_pred in eta_list:
                
        #model scores  
        R2 = r2_score(y_visc, y_pred)
        EV = explained_variance_score(y_visc, y_pred)
        MAE = mean_absolute_error(y_visc, y_pred)

        MSE = mean_squared_error(y_visc, y_pred)
        RMSE = np.sqrt(MSE)

        # add results to lists for a final comparison
               
        MAE_scores.append(round(MAE, 2))
        ExplVar_scores.append(round(EV, 2))
        RMSE_scores.append(round(RMSE, 2))
        R2_scores.append(round(R2, 4))
        
    
    #create table to compare MLA metrics
    MLA_columns = ['Model Name', 'Explained Variance','MAE', 'RMSE', 'R2']
    zippedList =  list(zip(MLA_names, ExplVar_scores, MAE_scores, RMSE_scores, R2_scores))
    df = pd.DataFrame(zippedList, columns = MLA_columns)#.sort_values(by=['R2'],ascending=False)
    return df


# In[12]:


eta_list = [eta_pow, eta_car, eta_yas, eta_cross ,eta_bin, eta_hb, eta_cas]
df_results = compare_models(eta_list)


# In[13]:


list_power =['K','n']
list_carreau = ['eta0',  'eta_inf','L', 'n']
list_yasuda = ['eta0', 'eta_inf', 'L', 'n','a']
list_cross = ['eta0', 'eta_inf', 'k', 'n']
list_bingham = ['yield_stress', 'plastic viscosity']
list_hb = ['yield_stress', 'K', 'n']
list_casson = ['yield_stress','casson_viscosity']

models_name_list = ['Ostwald – de Waele power law', 'Carreau', 'Carreau-Yasuda', 'Cross', 'Bingham',"Herschel-Bulkley", 'Casson']
coefficient_list = [list_power, list_carreau, list_yasuda, list_cross, list_bingham, list_hb, list_casson]


# In[14]:


df_final = df_ares

for i in range(len(models_name_list)):
    name = "%s_viscosity" % models_name_list[i]
    df_final[name] = eta_list[i]

df_final


# ## 5. Output files

# In[15]:


output_name = input("Enter output file name: ")
print('output file:',output_name,'_output.xlsx')
print('you can find it in the same folder containing this script and the input data file')


# In[16]:


l1 = NonNewtonian_param_list
l2 = yield_stress_param_list

parameters_values = [*l1, *l2]#.append()
parameters_values


# In[17]:


param_zippedList =  list(zip(models_name_list, coefficient_list, parameters_values))
df_param = pd.DataFrame(param_zippedList, columns = ['model','coefficient','value'])
df_param


# In[18]:


df_results = pd.merge(df_param, df_results, left_on='model', right_on='Model Name')
df_results = df_results.drop(['Model Name'], axis=1)
df_results


# In[19]:


# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter("%s_output.xlsx" % output_name, engine='xlsxwriter')


# In[20]:


# Convert the dataframe to an XlsxWriter Excel object.
df_final.to_excel(writer, sheet_name='data')
df_results.to_excel(writer, sheet_name='results')

# Close the Pandas Excel writer and output the Excel file.

writer.save()
