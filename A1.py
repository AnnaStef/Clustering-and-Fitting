import pandas as pd
import matplotlib.pyplot as plt
from numpy import arange
from matplotlib import pyplot
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# reading the files
def Read_data(file_name):
    """
    This function loads data from  excel file.
    Parameters:
    file_name (str): The path to the excel file containing the data.
   
    Returns: data and transpose of the data.
    """
    data1=pd.read_excel(file_name)
    data1=data1.iloc[[12],[39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58]] 
    data1.reset_index(drop=True, inplace=True)
    print(data1)
    data1_t=data1.T
    print(data1_t)
    return data1, data1_t

cdata, cdata_t = Read_data("C:\\Users\\shobi\\OneDrive\\Desktop\\Anna\\CO2 emission.xlsx")
pdata, pdata_t = Read_data("C:\\Users\\shobi\\OneDrive\\Desktop\\Anna\\ADS\\population.xlsx")

#Concatinating the 2 columns.
merged_data = pd.concat([cdata, pdata ])
merged_data=merged_data.set_axis(['CO2 Emission of UAE','Population of UAE'],axis=0,inplace=False)
merged_data=merged_data.set_axis(['Year','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014'],axis=1,inplace=False)
merged_data=merged_data.T
print(merged_data.describe())
merged_data.to_csv('merged_df.csv', index=False)
print(merged_data)


#Create Scatter Matrix.
pd.plotting.scatter_matrix(merged_data , figsize=(9.0, 9.0))
plt.tight_layout() # helps to avoid overlap of 
plt.title('Scatter Matrix')
plt.show()

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3)
merged_data['cluster'] = kmeans.fit_predict(merged_data[['Population of UAE','CO2 Emission of UAE']])
# Plot cluster membership
plt.scatter(merged_data['Population of UAE'], merged_data['CO2 Emission of UAE'], c=merged_data['cluster'])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='.', s=500)
plt.xlabel('Population of UAE')
plt.ylabel('CO2 Emission of UAE')
plt.title('Clustered Data')
plt.show()

#Perform fitting using equation.
def objective(x,a,b):
    return a*x+b

x ,y = merged_data['Population of UAE'], merged_data['CO2 Emission of UAE']

uae, _ = curve_fit(objective, x, y)
a,b = uae
print('y=%.5f * x+ %.5f' %(a,b))
pyplot.scatter(x,y)
x_line = arange(min(x),max(x),1)
y_line = objective(x_line,a,b)
pyplot.plot(x_line, y_line, color='red', linewidth=1)
plt.xlabel('CO2 Emission of UAE')
plt.ylabel('Population of UAE')
pyplot.title('Fitted Data')
pyplot.show() 


def err_range(data,col_name, col_name2):
    mean = merged_data[col_name2].mean()
    std_dev = data[col_name].std()
    upper_lim = mean + 2*std_dev
    lower_lim = mean - 2*std_dev
    return upper_lim, lower_lim

# Estimate upper and lower limits for a column named 'column_name'
upper_lim, lower_lim = err_range(merged_data, 'CO2 Emission of UAE', 'CO2 Emission of UAE')
print("Upper limit of CO2 Emission in UAE: ", upper_lim)
print("Lower limit of CO2 Emission in UAE: ", lower_lim)

upper_lim, lower_lim = err_range(merged_data, 'Population of UAE', 'Population of UAE')
print("Upper limit of CO2 population in UAE: ", upper_lim)
print("Lower limit of CO2 population in UAE: ", lower_lim)
