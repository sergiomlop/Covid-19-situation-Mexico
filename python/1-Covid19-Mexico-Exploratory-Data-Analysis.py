##### Import packages
# Basic packages
import pandas as pd
import numpy as np

# Data Visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import FeatureGroup, LayerControl, Map

# Other packages
import time
from datetime import datetime
import json
import geopandas as gpd

# To avoid warnings
import warnings
warnings.filterwarnings("ignore")





##### Functions

# Create labels on charts
def autolabel(plot):
    for p in plot.patches:
        plot.annotate(format(p.get_height(), '.0f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')  
# Plot a bar chart        
def graph_bar(data,title):        
    plt.figure(figsize=(35, 10))
    plot = sns.barplot(x = data.index, y = data.values, palette="rocket")
    plt.title(title)
    plt.xticks(rotation=90)
    autolabel(plot)
    plt.tight_layout()
    plt.show()

# Plot a pie chart    
def graph_pie(dictionary,title):
    dictionary = dict(sorted(dictionary.items(), key = lambda item: item[1], reverse=True))
    plt.figure(figsize=(10, 10))
    plt.pie(dictionary.values(), labels = dictionary.keys(), explode = [0.1 for i in range(len(dictionary.values()))],
            autopct='%.1f%%', shadow = True, labeldistance = 1.07, startangle = 45)
    plt.title(title)
    centre_circle = plt.Circle((0,0),0.80,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# # Initial analysis




##### Import data
# Check the csv's path before running it

df_est = pd.read_csv("CoordEstados.csv", encoding = "ISO-8859-1") # Mexican states data
df_cov = pd.read_csv("14.11.20 - COVID19MEXICO.csv", encoding = "ISO-8859-1") # Covid-19 data





##### Brief statistic of each DataFrame

# Mexican states data
print(' Mexican states data '.center(35,'#'))
df_est.info()
df_est.describe()
print(' ')

# Covid-19 data
print(' Covid-19 data '.center(35,'#'))
df_cov.info()
df_cov.describe()





##### Number of missing values for each DataFrame

for dataframe,name in zip([df_est,df_cov],['Mexican states data','Covid-19 data']):
    print(f' {name}  nº of NaNs: '.center(55,'#'))
    if dataframe.isna().sum().sum() == 0:
        print('0 NaNs in total dataset')
    else:
        for column in dataframe.columns:
            if dataframe[column].isna().sum() != 0:
                print(f'{column} has {round(dataframe[column].isna().sum()/dataframe.shape[0],3)}% of NaNs')


# # Exploratory Data Analysis




##### Transform all the mexican states from number to name

states_num = sorted(df_cov['ENTIDAD_RES'].unique())
states_name = df_est['Estado'].loc[:31].values

dict_states = dict()
for num,name in zip(states_num,states_name):
    dict_states[num] = name
print(dict_states)

df_cov['ENTIDAD_RES'] = df_cov['ENTIDAD_RES'].map(dict_states)





##### Transform all the mexican National Health System institutions from number to name

dict_sector = {1:'CRUZ ROJA', 2:'DIF',3:'ESTATAL',4:'IMSS',5:'IMSS-BIENESTAR',6:'ISSSTE',7:'MUNICIPAL',8:'PEMEX',
               9:'PRIVADA',10:'SEDENA',11:'SEMAR',12:'SSA',13:'UNIVERSITARIO',99:'NO ESPECIFICADO'}

df_cov['SECTOR'] = df_cov['SECTOR'].map(dict_sector)





##### Create a new colum with the time difference between been positive in COVID-19 and die
# If the person didn't die, time difference is 0

df_cov['FECHA_SINTOMAS'] = df_cov['FECHA_SINTOMAS'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_cov['FECHA_DEF'] = df_cov['FECHA_DEF'].replace('9999-99-99', '2001-01-01')

df_cov['FECHA_DEF'] = df_cov['FECHA_DEF'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_cov['DIFERENCIA'] = df_cov['FECHA_DEF'].sub(df_cov['FECHA_SINTOMAS'], axis=0)

df_cov['DIFERENCIA'] = df_cov['DIFERENCIA'] / np.timedelta64(1, 'D')
df_cov.loc[df_cov['DIFERENCIA']<0,'DIFERENCIA'] = 0


# ## Positive COVID-19 cases




##### Analysis of total number of positive cases
# Catalogo and Descriptores show all the information about the variables. 
# CLASIFICACION_FINAL has 1, 2 or 3 when a person is COVID-19 confirmed.

df_positive_cases =  df_cov[df_cov['CLASIFICACION_FINAL'].isin([1,2,3])]

fig, axes = plt.subplots(1, 4, figsize=(25, 5))

# 1st graph - Gender of positive cases
axes[0].set_title('Gender of positive cases')
plot = sns.barplot(x = ['Women','Men'],y = df_positive_cases['SEXO'].value_counts().values, palette = 'pastel', ax = axes[0])
plot.set(ylim = (450000,550000), ylabel = None, yticklabels = [])
plot.tick_params(left=False)
autolabel(plot)

# 2nd graph - Result of diagnostic (positive or not)
axes[1].set_title('Result of diagnostic')
pos, neg = [df_positive_cases.shape[0], df_cov.query('CLASIFICACION_FINAL == 7').shape[0]]
plot = sns.barplot(x = ['Positive','Negatuve'],y = [pos,neg], palette = 'pastel', ax = axes[1])
plot.set(ylim = (800000,1500000), ylabel = None, yticklabels = [])
plot.tick_params(left=False)
autolabel(plot)

# 3rd graph - Age distribution of positives cases
axes[2].set_title('Age distribution of positive cases')
plot = sns.distplot(df_positive_cases.EDAD, ax = axes[2])
plot.set(ylabel = None, yticklabels = [])
plot.tick_params(left=False)

# 4th graph - National Health System institution that provided the care
axes[3].set_title('National Health System institution \n that provided the care')
data = df_positive_cases.SECTOR.value_counts()
plot = sns.barplot(x = data.index[:5], y = data.values[:5],
                   palette = 'pastel', ax = axes[3])
plot.set(ylim = (10000,650000), ylabel = None, yticklabels = [])
plot.tick_params(left=False)
autolabel(plot)

plt.tight_layout()
plt.show()





##### Total number of positive cases by state

# Data Extraction
print(f' Total number of positive cases = {df_positive_cases.shape[0]} | % of total = {(df_positive_cases.shape[0]/df_cov.shape[0]):.3}% '.center(100,'#'))
data = df_positive_cases.groupby('ENTIDAD_RES').count()['CLASIFICACION_FINAL'].sort_values(ascending = False)

# Data Visualization
graph_bar(data, 'Total number of positive cases by state')





##### Mean age of positive cases by state

# Data Extraction
print(f' Total mean age of positive cases = {(df_positive_cases.EDAD.mean()):.3} '.center(100,'#'))
data = df_positive_cases.groupby('ENTIDAD_RES').mean()['EDAD'].sort_values(ascending = False)

# Data Visualization
graph_bar(data, 'Mean age of positive cases by state')





##### Common illness on total positive cases 

# Data Extraction
ill_name = ['DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA']
dict_ill_pos = dict()
for name in ill_name:
    dict_ill_pos[name] = df_positive_cases.query(f'{name} == 1').shape[0]
print(f' Most common illness on total positive cases  = {max(dict_ill_pos, key=dict_ill_pos.get)} '.center(100,'#'))

# Data Visualization
graph_pie(dict_ill_pos,'Common illness on total positive cases')





##### Common illness on total positive cases by state

for state in sorted(df_cov.ENTIDAD_RES.unique()):
    print(f' {state} '.center(100,'#'))
    # Data Extraction
    ill_name = ['DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA']
    dict_ill_pos = dict()
    for name in ill_name:
        dict_ill_pos[name] = df_positive_cases.query(f'{name} == 1 & ENTIDAD_RES == "{state}"').shape[0]
    print(f'Most common illness on total positive cases at {state} = {max(dict_ill_pos, key=dict_ill_pos.get)}')
    # Data Visualization
    graph_pie(dict_ill_pos,f'Common illness on total positive cases at {state}')


# ## Deceased COVID-19 cases




##### Analysis of total number of deceased cases
# FECHA_DEF has 9999-99-99 when a person COVID-19 confirmed. didn't die.

df_deceased_cases =  df_cov[df_cov['FECHA_DEF'] !='9999-99-99']

fig, axes = plt.subplots(1, 3, figsize=(25, 5))

# 1st graph - Gender of positive cases
axes[0].set_title('Gender of deceased cases')
plot = sns.barplot(x = ['Women','Men'],y = df_deceased_cases['SEXO'].value_counts().values, palette = 'pastel', ax = axes[0])
plot.set(ylim = (1200000,1400000), ylabel = None, yticklabels = [])
plot.tick_params(left=False)
autolabel(plot)

# 2nd graph - Age distribution of deceased cases
axes[1].set_title('Age distribution of deceased cases')
plot = sns.distplot(df_deceased_cases.EDAD, ax = axes[1])
plot.set(ylabel = None, yticklabels = [])
plot.tick_params(left=False)

# 3rd graph - Time between FECHA_SINTOMA and FECHA_DEF when a person die of COVID-19
axes[2].set_title('Days between been positive \n in COVID-19 and die')
data = df_cov['DIFERENCIA'].value_counts()
plot = sns.barplot(x = data.index[1:15], y = data.values[1:15], 
                   palette = 'pastel', order = data.index[1:15], ax = axes[2])
plot.set(ylim = (None,10000), ylabel = None, yticklabels = [])
plot.tick_params(left=False)
autolabel(plot)

plt.tight_layout()
plt.show()





##### Total number of deceased cases by state

# Data Extraction
print(f' Total number of deceased cases = {df_deceased_cases.shape[0]} | % of total = {(df_deceased_cases.shape[0]/df_cov.shape[0]):.3}% '.center(100,'#'))
data = df_deceased_cases.groupby('ENTIDAD_RES').count()['CLASIFICACION_FINAL'].sort_values(ascending = False)

# Data Visualization
graph_bar(data, 'Total number of deceased cases by state')





##### Mean age of deceased cases by state

# Data Extraction
print(f' Total mean age of deceased cases = {(df_deceased_cases.EDAD.mean()):.3} '.center(100,'#'))
data = df_deceased_cases.groupby('ENTIDAD_RES').mean()['EDAD'].sort_values(ascending = False)

# Data Visualization
graph_bar(data, 'Mean age of deceased cases by state')





##### Common illness on total deceased cases 

# Data Extraction
ill_name = ['DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA']
dict_ill_dec = dict()
for name in ill_name:
    dict_ill_dec[name] = df_deceased_cases.query(f'{name} == 1').shape[0]
print(f' Most common illness on total deceased cases  = {max(dict_ill_dec, key=dict_ill_dec.get)} '.center(100,'#'))

# Data Visualization
graph_pie(dict_ill_dec,'Common illness on total deceased cases')





##### Common illness on total positive cases by state

for state in sorted(df_cov.ENTIDAD_RES.unique()):
    print(f' {state} '.center(100,'#'))
    # Data Extraction
    ill_name = ['DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA']
    dict_ill_dec = dict()
    for name in ill_name:
        dict_ill_dec[name] = df_deceased_cases.query(f'{name} == 1 & ENTIDAD_RES == "{state}"').shape[0]
    print(f'Most common illness on total positive cases at {state} = {max(dict_ill_dec, key=dict_ill_dec.get)}')
    # Data Visualization
    graph_pie(dict_ill_dec,f'Common illness on total positive cases at {state}')


# # Interactive map




##### Replace the name of the states in the DataFrame with the json -- to avoid missing information

with open("mexico22.json") as f:
    data = json.load(f)
    
states_json = list()
for i in range(32):
    states_json.append(data['features'][i]['properties']['name'])
    
states_json = sorted(states_json)
states_df = sorted(df_cov.ENTIDAD_RES.unique())
print('In json the differences appear as:',sorted(set(states_json) - set(states_df)))
print('While in the DataFrame the differences appear as:',sorted(set(states_df) - set(states_json)))

dict_states = dict()
for json,df in zip(sorted(set(states_json) - set(states_df)),sorted(set(states_df) - set(states_json))):
    dict_states[df] = json
    
df_cov['ENTIDAD_RES'] = df_cov['ENTIDAD_RES'].replace(dict_states)





##### Number of confirmed cases and deceased cases on Mexico (dataset)

data = gpd.read_file("mexico22.json").sort_values('name',ascending = True).reset_index(drop = True)
data.rename(columns = {'name': "States"}, inplace=True)
data['Positives'] = df_cov[(df_cov['CLASIFICACION_FINAL'].isin([1,2,3])) & (~(df_cov['ENTIDAD_NAC'].isin([97,98,99])))].groupby('ENTIDAD_RES')                     .count()['CLASIFICACION_FINAL'].values
data['Deaths'] = df_cov[(~(df_cov['FECHA_DEF'] =='2001-01-01')) & (~(df_cov['ENTIDAD_NAC'].isin([97,98,99])))].groupby('ENTIDAD_NAC')                  .count()['CLASIFICACION_FINAL'].values
data





##### Number of confirmed cases on Mexico (Map)
# Check the json's path before running it

# Creation of the map
mexico_map = folium.Map(location=[23.634501, -102.552784], zoom_start=5.5, tiles = None)
folium.TileLayer('http://tile.stamen.com/watercolor/{z}/{x}/{y}.png', name = "Watercolor map", control = False, attr = "toner-bcg").add_to(mexico_map)
mexico_geo = r"mexico22.json"

# Adding confirmed cases layer
choropleth = folium.Choropleth(
    name='Confirmed cases',
    geo_data = mexico_geo,
    data = data,
    columns = ['States','Positives'],
    key_on = 'feature.properties.name',
    fill_color = 'YlOrRd', 
    fill_opacity = 0.65, 
    line_opacity = 0.5,
    threshold_scale = list(data['Positives'].quantile([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1])),
    overlay= False)

for key in choropleth._children:
    if key.startswith('color_map'):
        del(choropleth._children[key])
choropleth.add_to(mexico_map)

# Adding deceased cases layer
choropleth = folium.Choropleth(
    name='Deceased cases',
    geo_data = mexico_geo,
    data = data,
    columns = ['States','Deaths'],
    key_on = 'feature.properties.name',
    fill_color = 'YlOrBr', 
    fill_opacity = 0.65, 
    line_opacity = 0.5,
    threshold_scale = list(data['Deaths'].quantile([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1])),
    overlay= False)

for key in choropleth._children:
    if key.startswith('color_map'):
        del(choropleth._children[key])
choropleth.add_to(mexico_map)

# Adding pop-up tooltips
style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}

highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

data_Geo = gpd.GeoDataFrame(data , geometry = data.geometry)

pop_up = folium.features.GeoJson(
    data_Geo,
    style_function = style_function, 
    control = False,
    highlight_function = highlight_function, 
    tooltip = folium.features.GeoJsonTooltip(
        fields=['States','Positives', 'Deaths'],
        aliases=['State: ','Number of COVID-19 confirmed cases: ', 'Number of COVID-19 deceased cases: '],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px;")))
mexico_map.add_child(pop_up)
mexico_map.keep_in_front(pop_up)

# To control the layers
folium.LayerControl(collapsed=False).add_to(mexico_map)

mexico_map

