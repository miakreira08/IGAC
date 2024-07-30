import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import folium
from shapely.geometry import Polygon
import io 
from io import BytesIO
import base64
from scipy.stats import chi2_contingency
import seaborn as sns
import requests
import zipfile
from shapely import wkt

st.title('Caracterización según indicador de dinamica inmobiliaria')

st.title("Dimensión Social - Tasa ruralidad")
st.write(" La tasa rural se calcula teniendo en cuenta la proyección poblacional calculada por el DANE para el año 2022")
base=pd.read_excel('https://raw.githubusercontent.com/miakreira08/IGAC/main/indicador_general.xlsx')
poblacion=pd.read_excel("https://raw.githubusercontent.com/miakreira08/IGAC/main/poblacion_2.xlsx")
poblacion=poblacion[['COD_DPTO','DEPARTAMENTO','COD_DPTO-MPIO','MUNICIPIO','AÑO','ÁREA GEOGRÁFICA','Total']]
poblacion=poblacion.rename(columns={'COD_DPTO-MPIO':'DIVIPOLA','AÑO':'ANO','ÁREA GEOGRÁFICA':'ZONA','Total':'Total_poblacion'})
poblacion=poblacion[poblacion['ANO']==2022]

poblacion['ZONA'].value_counts()
mapping = {
    'Cabecera Municipal': 'URBANO',
    'Centros Poblados y Rural Disperso': 'RURAL'}
poblacion['ZONA'] = poblacion['ZONA'].replace(mapping)
df=poblacion.pivot_table(index=['DIVIPOLA', 'ANO'], columns='ZONA', values='Total_poblacion').reset_index()
df['tasa_rural']=df['RURAL']/df['Total']
base=base.merge(df[['DIVIPOLA','tasa_rural']],how='left',on=['DIVIPOLA'])
base=pd.read_excel("https://raw.githubusercontent.com/miakreira08/IGAC/main/base_con_geometry.xlsx")
bins = [0, 0.5, 0.8, 1]
labels = ['Menor a 50%', 'Entre 50% y 80%', 'Mayor al 80%']
base['categoria_tasa'] = pd.cut(base['tasa_rural'], bins=bins, labels=labels, right=False)

if 'geometry' not in base.columns:
    st.error("La columna 'geometry' no se encuentra en el DataFrame 'base'.")
else:
    # Convertir el DataFrame base a un GeoDataFrame
    base['geometry'] = base['geometry'].apply(lambda x: wkt.loads(x) if pd.notnull(x) else None)
    gdf = gpd.GeoDataFrame(base, geometry='geometry')

    # Establecer el CRS si no está ya establecido
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)  # Puedes cambiar "EPSG:4326" por el CRS adecuado

municipios_unicos = sorted(gdf['MPIO_CNMBR'].unique())
municipio_seleccionado = st.selectbox('Selecciona un municipio:',
                                      municipios_unicos)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
gdf.plot(column='categoria_tasa', ax=ax1, legend=True,
         legend_kwds={'title': 'Categoría de Tasa Rural', 'loc': 'upper left'})
municipio_resaltado = gdf[gdf['MPIO_CNMBR'] == municipio_seleccionado]
municipio_resaltado.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=2)
ax1.set_title('Mapa de Categoría de Tasa Rural')
ax1.set_axis_off()  # Ocultar ejes
gdf.plot(column='gmm_categoria', ax=ax2, legend=True,
         legend_kwds={'title': 'Indicador de Dinámica', 'loc': 'upper left'})
municipio_resaltado = gdf[gdf['MPIO_CNMBR'] == municipio_seleccionado]
municipio_resaltado.plot(ax=ax2, facecolor='none', edgecolor='red', linewidth=2)
ax2.set_title('Mapa de Categoría de GMM')
ax2.set_axis_off()  # Ocultar ejes

# Ajustar y mostrar los mapas
plt.tight_layout()
st.pyplot(fig)
st.subheader('Número de Municipios según el indicador de dinamica y Categoría de Tasa Rural')
pivot_table = pd.pivot_table(gdf,
                             values='MPIO_CNMBR',
                             index='gmm_categoria',
                             columns='categoria_tasa',
                             aggfunc='count',
                             fill_value=0)

desired_column_order = ['Menor a 50%', 'Entre 50% y 80%', 'Mayor al 80%']
pivot_table = pivot_table.reindex(columns=desired_column_order)

# Ordenar las filas
desired_row_order = ['alto', 'medio', 'baja']
pivot_table = pivot_table.reindex(desired_row_order)

st.table(pivot_table)
desired_column_order = ['Menor a 50%', 'Entre 50% y 80%', 'Mayor al 80%']
desired_row_order = ['alto', 'medio', 'baja']

# Convertir las columnas a categóricas con el orden deseado
gdf['categoria_tasa'] = pd.Categorical(gdf['categoria_tasa'], categories=desired_column_order, ordered=True)
gdf['gmm_categoria'] = pd.Categorical(gdf['gmm_categoria'], categories=desired_row_order, ordered=True)

contingency_table = pd.crosstab(gdf['categoria_tasa'], gdf['gmm_categoria'])

# Realizar la prueba de chi-cuadrado
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Configurar Streamlit
st.subheader('Prueba de Chi-Cuadrado de Independencia')

# Mostrar los resultados de la prueba de chi-cuadrado
st.write(f'Chi-cuadrado: {chi2:.2f}')
st.write(f'p-valor: {p:.4f}')
st.write(f'Grados de libertad: {dof}')

# Interpretación del resultado
st.subheader('Interpretación del Resultado')
if p < 0.05:
    st.write("Existe una relación significativa entre las variables 'categoria_tasa' y 'gmm_categoria'.")
else:
    st.write("No existe una relación significativa entre las variables 'categoria_tasa' y 'gmm_categoria'.")

# Visualización de la relación
st.subheader('Visualización de la Relación')

# Heatmap de las frecuencias observadas

fig, ax = plt.subplots()
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
ax.set_title('Frecuencias Observadas')
plt.show()
st.pyplot(fig)

pivot_long = pivot_table.reset_index().melt(id_vars='gmm_categoria', value_vars=desired_column_order, var_name='categoria_tasa', value_name='count')

# Crear el gráfico de barras dobles
import plotly.express as px
fig = px.bar(pivot_long, x='categoria_tasa', y='count', color='gmm_categoria', barmode='group',
             category_orders={'categoria_tasa': desired_column_order, 'gmm_categoria': desired_row_order})

# Configurar el diseño del gráfico
fig.update_layout(
    title="Diagrama de Barras Dobles de categoria_tasa y gmm_categoria",
    xaxis_title="Categoria Tasa",
    yaxis_title="Cantidad",
    legend_title="Indicador de dinamica inmobiliaria"
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

st.title('Dimensión social - Indice de pobreza multidimensional')

st.write("Para este análisis es importante tener en cuenta que según el DANE se consideran pobres"
         "a aquellos hogares con un IPM igual o superior al 33%")
pobreza=pd.read_excel("C:/Users/1016111808/Downloads/anexo-censal-pobreza-municipal-2018.xlsx")
pobreza=pobreza.rename(columns={'ID':'DIVIPOLA','IPM Municipal':'IPM'})
gdf=gdf.merge(pobreza[['DIVIPOLA','IPM']],how='left',on='DIVIPOLA')

gdf['pobre_cat']=gdf['IPM'].apply(lambda x: "pobre" if x>=33 else "no pobre")

fig, ax = plt.subplots(figsize=(12, 10))

# Crear el mapa categórico para la variable 'pobre_cat'
gdf.plot(column='pobre_cat', ax=ax, legend=True, cmap='coolwarm', edgecolor='black')

# Configuraciones adicionales del mapa
ax.set_title('Mapa de Pobreza por Municipio')
ax.set_axis_off()  # Ocultar ejes

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

fig, ax = plt.subplots()
gdf.plot.scatter(x='IPM', y='pca_nuevo', ax=ax, color='blue', alpha=0.5)

# Configuraciones adicionales del gráfico
ax.set_title('Dispersión entre IPM e indicador de dinamica inmobiliaria')
ax.set_xlabel('IPM')
ax.set_ylabel('Indicador de dinamica inmobiliaria')

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

correlacion = gdf['IPM'].corr(gdf['pca_nuevo'])

# Configurar Streamlit

# Mostrar la medida de correlación
st.subheader('Medida de Correlación')
st.write(f'La correlación entre IPM y PCA Nuevo es: {correlacion:.2f}')

st.subheader('Prueba de chi-cuadrado')
gdf['gmm_categoria'] = pd.Categorical(gdf['gmm_categoria'], categories=desired_row_order, ordered=True)

contingency_table = pd.crosstab(gdf['pobre_cat'], gdf['gmm_categoria'])

# Realizar la prueba de chi-cuadrado
chi2, p, dof, expected = chi2_contingency(contingency_table)

st.write(contingency_table)
# Configurar Streamlit
st.subheader('Prueba de Chi-Cuadrado de Independencia')

# Mostrar los resultados de la prueba de chi-cuadrado
st.write(f'Chi-cuadrado: {chi2:.2f}')
st.write(f'p-valor: {p:.4f}')
st.write(f'Grados de libertad: {dof}')

# Interpretación del resultado
st.subheader('Interpretación del Resultado')
if p < 0.05:
    st.write("Existe una relación significativa entre las variables 'categoria_tasa' y 'gmm_categoria'.")
else:
    st.write("No existe una relación significativa entre las variables 'categoria_tasa' y 'gmm_categoria'.")

pivot_table = pd.pivot_table(gdf,
                             values='MPIO_CNMBR',
                             index='gmm_categoria',
                             columns='pobre_cat',
                             aggfunc='count',
                             fill_value=0)

# Ordenar las filas
desired_row_order = ['alto', 'medio', 'baja']
desired_column_order=['pobre','no pobre']
pivot_table = pivot_table.reindex(desired_row_order)

pivot_long = pivot_table.reset_index().melt(id_vars='gmm_categoria', value_vars=desired_column_order, var_name='pobre_cat', value_name='count')

fig = px.bar(pivot_long, x='pobre_cat', y='count', color='gmm_categoria', barmode='group',
             category_orders={'gmm_categoria': desired_row_order})

# Configurar el diseño del gráfico
fig.update_layout(
    title="Diagrama de Barras Dobles de pobreza y gmm_categoria",
    xaxis_title="Pobreza",
    yaxis_title="Cantidad",
    legend_title="Indicador de dinamica inmobiliaria"
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

st.subheader("Dimensión social - Servicios_publicos")
servicios=pd.read_excel("C:/Users/1016111808/Downloads/20240710_TP_PEI_Servicios_SUI_V2.xlsx")
servicios=servicios[servicios['AÑO']==2022]
servicios=servicios.groupby('DIVIPOLA').agg({'Numero de suscriptores':'sum',
                                   'Consumo total':'sum'})
servicios=servicios.reset_index()
servicios=servicios.rename(columns={'Numero de suscriptores':'suscriptores',
                          'Consumo total':'Consumo'})
gdf=gdf.merge(servicios,how='left',on='DIVIPOLA')
gdf.columns
fig, ax = plt.subplots(figsize=(12, 10))

gdf=gpd.GeoDataFrame(gdf)
# Crear el mapa categórico para la variable 'pobre_cat'
gdf.plot(column='Consumo', ax=ax, legend=True, cmap='coolwarm', edgecolor='black')

# Configuraciones adicionales del mapa
ax.set_title('Mapa de consumo de servicios publicos para el año 2022 ')
ax.set_axis_off()  # Ocultar ejes

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

percentile_33 = gdf['Consumo'].quantile(0.3333)
percentile_66 = gdf['Consumo'].quantile(0.6666)

def categorize_consumo(value):
    if value <= percentile_33:
        return 'Bajo'
    elif value <= percentile_66:
        return 'Medio'
    elif value >percentile_66:
        return 'Alto'
    else:
        return 'Bajo'

gdf['Consumo_Categoria'] = gdf['Consumo'].apply(categorize_consumo)
gdf['Consumo_Categoria'] = pd.Categorical(gdf['Consumo_Categoria'], categories=['Bajo', 'Medio', 'Alto'], ordered=True)
from matplotlib.colors import ListedColormap

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Ploteo del mapa con la categoría
gdf.plot(column='Consumo_Categoria', ax=ax, legend=True, cmap='viridis', edgecolor='black')

# Configuraciones adicionales del mapa
ax.set_title('Mapa de consumo de servicios publicos por Municipio')
ax.set_axis_off()  # Ocultar ejes

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

contingency_table = pd.crosstab(gdf['Consumo_Categoria'], gdf['gmm_categoria'])

# Realizar la prueba de chi-cuadrado
chi2, p, dof, expected = chi2_contingency(contingency_table)

st.write(contingency_table)
# Configurar Streamlit
st.subheader('Prueba de Chi-Cuadrado de Independencia')

# Mostrar los resultados de la prueba de chi-cuadrado
st.write(f'Chi-cuadrado: {chi2:.2f}')
st.write(f'p-valor: {p:.4f}')
st.write(f'Grados de libertad: {dof}')

# Interpretación del resultado
st.subheader('Interpretación del Resultado')
if p < 0.05:
    st.write("Existe una relación significativa entre las variables 'Consumo_categoria' y 'dinamica'.")
else:
    st.write("No existe una relación significativa entre las variables 'Consumo_categoria' y 'dinamica'.")

pivot_table = pd.pivot_table(gdf,
                             values='MPIO_CNMBR',
                             index='gmm_categoria',
                             columns='Consumo_Categoria',
                             aggfunc='count',
                             fill_value=0)

# Ordenar las filas
desired_row_order = ['alto', 'medio', 'baja']
desired_column_order=['Alto','Medio','Bajo']
pivot_table = pivot_table.reindex(desired_row_order)
pivot_long = pivot_table.reset_index().melt(id_vars='gmm_categoria', value_vars=desired_column_order, var_name='Consumo_Categoria', value_name='count')

fig = px.bar(pivot_long, x='Consumo_Categoria', y='count', color='gmm_categoria', barmode='group',
             category_orders={'gmm_categoria': desired_row_order})

# Configurar el diseño del gráfico
fig.update_layout(
    title="Diagrama de Barras Dobles de Consumo de servicios publicos y gmm_categoria",
    xaxis_title="Consumo servicios publicos",
    yaxis_title="Cantidad",
    legend_title="Indicador de dinamica inmobiliaria"
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

st.subheader("Análisis de correspondencias multiples para servicios publicos")
import prince
Y=gdf[['gmm_categoria','categoria_tasa','Consumo_Categoria','pobre_cat']]
mca = prince.MCA(
    n_components=2,
    n_iter=15,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
)
mca = mca.fit(Y)
one_hot = pd.get_dummies(Y)
mca_no_one_hot = prince.MCA(one_hot=False)
mca_no_one_hot = mca_no_one_hot.fit(one_hot)

st.write(mca.eigenvalues_summary)

mca_results = mca.transform(Y)
row_coords = mca.row_coordinates(Y)
col_coords = mca.column_coordinates(Y)
from adjustText import adjust_text
sns.set(style="whitegrid")
Etiqueta= [
        'Alta', 'Media', 'Baja',
        'Menor a 50%', 'Entre 50% y 80%', 'Mayor al 80%',
        'Bajo', 'Medio', 'Alto',
        'no pobre', 'pobre']
Columna =[
        'GMM', 'GMM', 'GMM',
        'Tasa', 'Tasa', 'Tasa',
        'Consumo', 'Consumo', 'Consumo',
        'Pobreza', 'Pobreza']
column_colors = {
    'GMM': 'blue',
    'Tasa': 'green',
    'Consumo': 'red',
    'Pobreza': 'purple'
}
col_coords['Etiqueta']=Etiqueta
col_coords['Columna']=Columna
col_coords['Color'] = col_coords['Columna'].map(column_colors)
# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(14, 10))
# Graficar las coordenadas de las columnas
scatter = ax.scatter(col_coords[0], col_coords[1], s=100, alpha=0.7, edgecolor='w', linewidth=0.5,c=col_coords['Color'])
# Añadir etiquetas a las columnas
texts = [ax.text(x, y, label, fontsize=12, ha='right', va='bottom') for x, y, label in zip(col_coords[0], col_coords[1],col_coords['Etiqueta'] )]
# Evitar la superposición de etiquetas
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
# Añadir etiquetas a los ejes
ax.set_xlabel('Dim 1', fontsize=14)
ax.set_ylabel('Dim 2', fontsize=14)
# Añadir título
ax.set_title('Coordenadas de las Columnas en MCA', fontsize=16)
# Añadir cuadrícula
ax.grid(True)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=column)
           for column, color in column_colors.items()]
ax.legend(handles=handles, title='Columnas')
# Mejorar el layout del gráfico
plt.tight_layout()
# Mostrar el gráfico en Streamlit
st.pyplot(fig)
plt.show()

from fpdf import FPDF
st.title()
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Streamlit Report", ln=True, align='C')
pdf.output("streamlit_report.pdf")
