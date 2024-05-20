#!/usr/bin/env python
# coding: utf-8

# # Estadísticas comerciales del proyecto: E-news Express
# 
# 

# ## Contexto empresarial
# 
# El advenimiento de los portales de noticias electrónicas nos ha ofrecido una gran oportunidad para obtener rápidamente actualizaciones sobre los eventos diarios que ocurren a nivel mundial. La información en estos portales se recupera electrónicamente de bases de datos en línea, se procesa utilizando una variedad de software y luego se transmite a los usuarios. Hay múltiples ventajas de transmitir noticias electrónicamente, como un acceso más rápido al contenido y la capacidad de utilizar diferentes tecnologías como audio, gráficos, video y otros elementos interactivos que no se utilizan o aún no son comunes en los periódicos tradicionales.
# 
# E-news Express, un portal de noticias en línea, tiene como objetivo expandir su negocio mediante la adquisición de nuevos suscriptores. Dado que cada visitante del sitio web realiza ciertas acciones en función de su interés, la compañía planea analizar estas acciones para comprender los intereses de los usuarios y determinar cómo impulsar un mejor compromiso. Los ejecutivos de E-news Express opinan que ha habido una disminución en los nuevos suscriptores mensuales en comparación con el año pasado porque la página web actual no está lo suficientemente bien diseñada en términos de esquema y contenido recomendado para mantener a los clientes comprometidos el tiempo suficiente para tomar la decisión de suscribirse.
# 
# [Las empresas suelen analizar las respuestas de los usuarios a dos variantes de un producto para decidir cuál de las dos variantes es más eficaz. Esta técnica experimental, conocida como prueba A/B, se utiliza para determinar si una nueva función atrae a los usuarios en función de una métrica elegida.]
# 
# 
# ## Objetivo
# 
# El equipo de diseño de la empresa investigó y creó una nueva página de destino que tiene un nuevo esquema y muestra contenido más relevante en comparación con la página anterior. Para probar la eficacia de la nueva página de destino para reunir nuevos suscriptores, el equipo de ciencia de datos realizó un experimento seleccionando al azar a 100 usuarios y dividiéndolos en dos grupos por igual. La página de destino existente se mostró al primer grupo (grupo de control) y la nueva página de destino al segundo grupo (grupo de tratamiento). Se recopilaron datos sobre la interacción de los usuarios de ambos grupos con las dos versiones de la página de destino. Como científico de datos en E-news Express, se le ha pedido que explore los datos y realice un análisis estadístico (a un nivel de significación del 5 %) para determinar la eficacia de la nueva página de destino para reunir nuevos suscriptores para el portal de noticias. respondiendo las siguientes preguntas:
# 
# 1. ¿Los usuarios pasan más tiempo en la nueva página de destino que en la página de destino existente?
# 
# 2. ¿La tasa de conversión (la proporción de usuarios que visitan la página de destino y se convierten) de la página nueva es mayor que la tasa de conversión de la página anterior?
# 
# 3. ¿El estado convertido depende del idioma preferido? [Sugerencia: cree una tabla de contingencia usando la función pandas.crosstab()]
# 
# 4. ¿El tiempo dedicado a la nueva página es el mismo para los diferentes usuarios de idiomas?
# 
# 
# ## Diccionario de datos
# 
# Los datos contienen información sobre la interacción de los usuarios de ambos grupos con las dos versiones de la página de destino.
# 
# 1. user_id - ID de usuario único de la persona que visita el sitio web
# 
# 2. grupo: si el usuario pertenece al primer grupo (control) o al segundo grupo (tratamiento)
# 
# 3. landing_page: si la página de destino es nueva o antigua
# 
# 4. tiempo_pasado_en_la_página: tiempo (en minutos) pasado por el usuario en la página de destino
# 
# 5. convertido: si el usuario se convierte en suscriptor del portal de noticias o no
# 
# 6. language_preferred: idioma elegido por el usuario para ver la página de destino

# ### **Lea atentamente las instrucciones antes de iniciar el proyecto.**
# Este es un archivo comentado de Jupyter IPython Notebook en el que se mencionan todas las instrucciones y tareas a realizar.
# * Se proporcionan espacios en blanco '_______' en el cuaderno que necesita ser llenado con un código apropiado para obtener el resultado correcto. Con cada espacio en blanco '_______', hay un comentario que describe brevemente lo que debe completarse en el espacio en blanco.
# * Identifique la tarea a realizar correctamente, y solo entonces proceda a escribir el código requerido.
# * Complete el código donde se le solicite en las líneas comentadas como "# escriba su código aquí" o "# complete el código". Ejecutar código incompleto puede arrojar un error.
# * Ejecute los códigos de forma secuencial desde el principio para evitar errores innecesarios.
# * Agregar los resultados/observaciones (cuando se mencionen) derivados del análisis en la presentación y enviar los mismos. Cualquier detalle matemático o computacional que sea una parte calificada del proyecto se puede incluir en la sección Apéndice de la presentación.

# ### Importar todas las bibliotecas necesarias

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo para las visualizaciones
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
get_ipython().run_line_magic('matplotlib', 'inline')

# Librerías para análisis estadístico
import scipy.stats as stats


# In[2]:


#version Scipy
import scipy
scipy.__version__


# ### Cargando el conjunto de datos

# In[3]:


# complete the code below to load the dataset
df = pd.read_csv('H:/Cursos/Data Science/Soluciones/M2Noticias exprés/abtest.csv')


# ## Explore el conjunto de datos y extraiga información mediante el análisis exploratorio de datos

# ### Resumen de datos

# Los pasos iniciales para obtener una visión general de cualquier conjunto de datos son:
# - observe las primeras filas del conjunto de datos, para verificar si el conjunto de datos se ha cargado correctamente o no
# - obtener información sobre el número de filas y columnas en el conjunto de datos
# - Averigüe los tipos de datos de las columnas para garantizar que los datos se almacenen en el formato preferido y que el valor de cada propiedad sea el esperado.
# - verifique el resumen estadístico del conjunto de datos para obtener una descripción general de las columnas numéricas de los datos

# #### Mostrar las primeras filas del conjunto de datos

# In[4]:


# Mostrar las primeras 5 filas del conjunto de datos
df.head()


# #### Mostrar las últimas filas del conjunto de datos

# In[5]:


# Mostrar las últimas 5 filas del conjunto de datos
df.tail()


# #### Comprobando la forma del conjunto de datos

# In[6]:


# view the shape of the dataset
df.shape


# * hay 6 columnas y 100 filas

# #### Comprobación de los tipos de datos de las columnas para el conjunto de datos

# In[7]:


# check the data types of the columns in the dataset
df.info()


# #### Comentarios:
# 
# * El conjunto de datos contiene 100 filas y 6 columnas.
# * Cada columna tiene 100 entradas no nulas, lo que indica que no hay valores faltantes en el conjunto de datos.
# * 'user_id' es de tipo int64, probablemente representando identificadores únicos para los usuarios.
# * 'group' es de tipo object, probablemente indicando diferentes grupos experimentales o de control.
# * 'landing_page' también es un object, sugiriendo que contiene datos categóricos sobre las páginas de destino con las que interactuaron los usuarios.
# * 'time_spent_on_the_page' es un float64, representando el tiempo que los usuarios pasaron en la página de destino en segundos u otra unidad de tiempo.
# * 'converted' es un object, indicando si el usuario completó una acción deseada, como realizar una compra o registrarse.
# * 'language_preferred' es un object, representando el idioma preferido del usuario.
# 

# In[8]:


#Cambio los datos a categoricos o objetos para un mejor analisis
for col in ['group', 'landing_page', 'converted', 'language_preferred']:
    df[col] = df[col].astype('category')
for col in ['user_id']:
    df[col] = df[col].astype('object')
df.info()


# In[9]:


cat_cols = ['group', 'landing_page', 'converted', 'language_preferred']
for column in cat_cols:
    print(df[column].value_counts())
    print("-" * 50)


# #### Obtener el resumen estadístico de las variables numéricas

# In[10]:


#Revisión del estadistico de datos
df.describe().T


# #### Obtener el resumen estadístico de las variables categóricas

# In[11]:


# write your code here to print the categorical summary statistics
df.describe(include = ["category"]).T


# ### Comprobar si faltan valores

# In[12]:


#  Valores faltantes
df.isnull().sum()


# In[13]:


#  Valores faltantes
df.isna().sum()


# * No exiten valores faltantes

# ### Buscar duplicados

# In[14]:


# Buscar duplicados   
df.duplicated().sum()


# In[15]:


# Quitar duplicados
#df.drop_duplicates(inplace= True)


# * No existen valores duplicados.

# # Analisis Exploratorio de Datos

# ### 1. Análisis univariado

# #### 1.1 Tiempo de permanencia en la página

# In[16]:


# Histograma
# Crear el histograma
hist_plot = sns.histplot(data=df, x='time_spent_on_the_page', bins=10)
plt.title('Histograma del Tiempo de Permanencia en la Página')
plt.xlabel('Tiempo de permanencia en la página (segundos)')
plt.ylabel('Frecuencia')
plt.show()

# Obtener los datos del histograma
counts, edges = hist_plot.patches[0].get_height(), hist_plot.patches[0].get_x()
hist_data = {
    'Intervalo (segundos)': [],
    'Frecuencia': []
}

for patch in hist_plot.patches:
    interval = f"{patch.get_x():.1f} - {patch.get_x() + patch.get_width():.1f}"
    frequency = patch.get_height()
    hist_data['Intervalo (segundos)'].append(interval)
    hist_data['Frecuencia'].append(frequency)

# Crear la tabla resumen
tabla_resumen = pd.DataFrame(hist_data)

# Mostrar la tabla resumen
print(tabla_resumen.to_string(index=False))
# --------------------
# Crear el gráfico de densidad (KDE)
kde_plot = sns.displot(data=df, x='time_spent_on_the_page', kind='kde')
plt.title('Gráfico de Densidad del Tiempo de Permanencia en la Página')
plt.xlabel('Tiempo de permanencia en la página (segundos)')
plt.ylabel('Densidad')
plt.show()

# Obtener los datos del gráfico de densidad (KDE)
line = kde_plot.ax.lines[0]
x_data = line.get_xdata()
y_data = line.get_ydata()

# Crear un DataFrame temporal con los datos de densidad
kde_data = pd.DataFrame({'Tiempo (segundos)': x_data, 'Densidad': y_data})

# Definir los intervalos de tiempo
kde_data['Intervalo'] = pd.cut(kde_data['Tiempo (segundos)'], bins=range(0, 13, 1), right=False)

# Calcular la densidad promedio para cada intervalo
kde_summary = kde_data.groupby('Intervalo')['Densidad'].mean().reset_index()

# Renombrar las columnas para mayor claridad
kde_summary.columns = ['Intervalo (segundos)', 'Densidad Promedio']

# Mostrar la tabla resumen
print(kde_summary.to_string(index=False))
# --------------------
# Crear el gráfico de caja (boxplot)
box_plot = sns.boxplot(data=df, x='time_spent_on_the_page')
plt.title('Boxplot del Tiempo de Permanencia en la Página')
plt.xlabel('Tiempo de permanencia en la página (segundos)')
plt.show()

# Obtener las estadísticas del boxplot
boxplot_stats = df['time_spent_on_the_page'].describe(percentiles=[.25, .5, .75])

# Calcular el rango intercuartílico (IQR) y los valores atípicos
Q1 = boxplot_stats['25%']
Q3 = boxplot_stats['75%']
IQR = Q3 - Q1
outliers = df['time_spent_on_the_page'][(df['time_spent_on_the_page'] < (Q1 - 1.5 * IQR)) | (df['time_spent_on_the_page'] > (Q3 + 1.5 * IQR))].tolist()

# Crear la tabla resumen con los datos del boxplot
tabla_resumen_boxplot = pd.DataFrame({
    'Métrica': [
        'Primer Cuartil (Q1)', 'Mediana', 'Tercer Cuartil (Q3)', 'Rango Intercuartílico (IQR)', 
        'Valor Mínimo', 'Valor Máximo', 'Valores Atípicos'
    ],
    'Descripción': [
        '25% de los usuarios debajo de este tiempo', '50% de los usuarios debajo de este tiempo (mediana)', 
        '75% de los usuarios debajo de este tiempo', 'Diferencia entre Q3 y Q1', 
        'Tiempo mínimo observado', 'Tiempo máximo observado', 'Tiempos fuera de los límites normales'
    ],
    'Valor': [
        Q1, boxplot_stats['50%'], Q3, IQR, boxplot_stats['min'], boxplot_stats['max'], outliers
    ]
})

# Mostrar la tabla resumen
print(tabla_resumen_boxplot.to_string(index=False))


# #### 1.2 Grupo

# In[17]:


df['group'].value_counts()


# In[18]:


sns.countplot(data=df,x='group')
plt.show()


# #### 1.3 Página de destino

# In[19]:


df['landing_page'].value_counts()


# In[20]:


# complete the code to plot the countplot
sns.countplot(data=df,x="landing_page")
plt.show()


# #### 1.4 Convertido

# In[21]:


df['converted'].value_counts()


# In[22]:


# complete the code to plot the countplot
sns.countplot(data=df,x="converted")
plt.show()


# #### 1.5 Idioma preferido

# In[23]:


df['language_preferred'].value_counts()


# In[24]:


# complete the code to plot the countplot
sns.countplot(data=df,x="language_preferred")
plt.show()


# ### 2. Análisis bivariado

# #### 2.1 Página de destino vs Tiempo de permanencia en la página

# In[25]:


sns.boxplot(data=df,x= 'landing_page',y= 'time_spent_on_the_page')
plt.show()


# In[26]:


plt.figure(figsize=(10,6))
# Calcular estadísticas descriptivas para cada página de destino
summary_stats = df.groupby('landing_page')['time_spent_on_the_page'].describe()

# Mostrar la tabla resumen
print(summary_stats)



# #### Observaciones: Promedio (mean):
# 
# * El tiempo promedio de permanencia en la nueva página (new) es de 6.2232 segundos.
# * El tiempo promedio de permanencia en la página antigua (old) es de 4.5324 segundos.
# * Observamos que los usuarios pasan en promedio más tiempo en la nueva página en comparación con la página antigua.

# #### 2.2 Estado de conversión frente al tiempo de permanencia en la página

# In[27]:


# complete el código para trazar un gráfico adecuado para comprender la relación entre las columnas 'time_spent_on_the_page' y 'converted'
sns.boxplot(data = df,x='converted',y='time_spent_on_the_page')
plt.show()


# In[28]:


# Calcular estadísticas descriptivas para cada página de destino
summary_stats = df.groupby('converted')['time_spent_on_the_page'].describe()

# Mostrar la tabla resumen
print(summary_stats)


# ### Observaciones
# * El tiempo promedio de permanencia de los usuarios que no se convirtieron es de 3.9159 segundos.
# * El tiempo promedio de permanencia de los usuarios que se convirtieron es de 6.6231 segundos.
# * Los usuarios que se convirtieron pasaron significativamente más tiempo en la página en comparación con los que no se convirtieron.

# #### 2.3 Idioma preferido frente al tiempo de permanencia en la página

# In[29]:


# write the code to plot a suitable graph to understand the distribution of 'time_spent_on_the_page' among the 'language_preferred'
sns.boxplot(data = df, x = "language_preferred", y = "time_spent_on_the_page")
plt.show()


# In[30]:


# Calcular estadísticas descriptivas para cada página de destino
summary_stats = df.groupby('language_preferred')['time_spent_on_the_page'].describe().T

# Mostrar la tabla resumen
print(summary_stats)


# #### Observaciones:
# Promedio de Tiempo de Permanencia:
# 
# * En promedio, los usuarios que prefieren el inglés pasan aproximadamente 5.56 segundos en la página.
# * Los usuarios que prefieren el francés tienen un tiempo de permanencia promedio de aproximadamente 5.25 segundos.
# * Los usuarios que prefieren el español tienen un tiempo de permanencia promedio de aproximadamente 5.33 segundos.

# ## 1. ¿Pasan los usuarios más tiempo en la nueva página de destino que en la página de destino existente?

# ### Realizar análisis visual

# In[31]:


# visual analysis of the time spent on the new page and the time spent on the old page
plt.figure(figsize=(8,6))
sns.boxplot(x = 'landing_page', y = 'time_spent_on_the_page', data = df)
plt.show()



# In[32]:


# Calcular estadísticas descriptivas para cada página de destino
summary_stats = df.groupby('landing_page')['time_spent_on_the_page'].describe().T

# Mostrar la tabla resumen
print(summary_stats)


# * Si los usuarios pasan mas tiempo en la pagina nueva, alrrededor de 6 minutos, mientras que en la vieja pagina 5 minutos

# # 3. Hipótesis probadas y resultados
# 

# ### Paso 1: Definir las hipótesis nula y alternativa

# $H_0$:  El tiempo medio de interacción con la nueva pagina es igual al tiempo medio de los usuarios en la pagina antigua.
# 
# $H_a$: El tiempo medio de interacción con la nueva pagina es mayor al tiempo medio de los usuarios en la pagina antigua.
# Entendiendo que $μ1$ es el tiempo medio de los usuarios en la pagina nueva y $μ2$  es el tiempo medio de los usuarios en la pagina antigua.
# 
# Matemáticamente, las hipótesis formuladas anteriormente se pueden escribir como:
# $H_0$:$μ1=μ2
# 
# $H_a$:$μ1>μ2

# ### Paso 2: Seleccion de la prueba adecuada

# Esta es una prueba de una cola sobre dos medias poblacionales de dos poblaciones independientes. Se desconocen las desviaciones estándar de la población. **Se tiene los datos de 2 medias de poblaciones independientes, con desviacion estandar desconocida Se hara uso de la "Prueba T"**.

# 
# ### Paso 3: Decidir el nivel de significancia

# Como se indica en el enunciado del problema, seleccionamos $\alpha = 0.05$.

# ### Paso 4: recopilar y preparar datos

# In[33]:


# create subsetted data frame for new landing page users 
time_spent_new = df[df['landing_page'] == 'new']['time_spent_on_the_page']

# create subsetted data frame for old landing page users
time_spent_old = df[df['landing_page'] == 'old']['time_spent_on_the_page'] ##Complete the code


# In[34]:


print('The sample standard deviation of the time spent on the new page is:', round(time_spent_new.std(),2))
print('The sample standard deviation of the time spent on the new page is:', round(time_spent_old.std(),2))


# **Con base en las desviaciones estándar de la muestra de los dos grupos, decida si se puede suponer que las desviaciones estándar de la población son iguales o desiguales**.

# ### Paso 5: Calcular el valor p

# In[35]:


# complete the code to import the required function
from scipy.stats import ttest_ind 

# write the code to calculate the p-value
test_stat, p_value =  ttest_ind(time_spent_new, time_spent_old, equal_var = False, alternative = 'greater')  #complete the code by filling appropriate parameters in the blanks

print('The p-value is', p_value)
print('The p-value is', p_value,"si es menor a la significancia $\alpha = 0.05$ se rechaza hipotesis nula.")


# ### Paso 6: Compare el valor p con $\alpha$

# In[36]:


# print the conclusion based on p-value
if p_value < 0.05:
    print(f'As the p-value {p_value} is less than the level of significance, we reject the null hypothesis.')
else:
    print(f'As the p-value {p_value} is greater than the level of significance, we fail to reject the null hypothesis.')


# ### Paso 7: Extraer inferencia

# Como el valor p (~0.00013) es menor que el nivel de significación, podemos rechazar la hipótesis nula. Por lo tanto, tenemos suficiente evidencia para respaldar la afirmación de que el tiempo medio de interacción con la nueva pagina es mayor al tiempo medio de los usuarios en la pagina antigua.

# 
# 
# 

# ## 2. ¿La tasa de conversión (la proporción de usuarios que visitan la página de destino y se convierten) de la página nueva es mayor que la tasa de conversión de la página anterior?

# ### Realizar análisis visual

# In[37]:


# Completar el código para comparar visualmente la tasa de conversión para la nueva página y la tasa de conversión para la página antigua
pd.crosstab(df['landing_page'], df['converted'], normalize='index').plot(kind="bar", figsize=(8,4), stacked=True)
plt.legend()
plt.show()


# ### Paso 1: Definir las hipótesis nula y alternativa

# $H_0$:  El tiempo medio de interacción con la nueva pagina es igual al tiempo medio de los usuarios en la pagina antigua.
# 
# $H_a$: El tiempo medio de interacción con la nueva pagina es mayor al tiempo medio de los usuarios en la pagina antigua.
# Entendiendo que $μ1$ es el tiempo medio de los usuarios en la pagina nueva y $μ2$  es el tiempo medio de los usuarios en la pagina antigua.
# 
# Matemáticamente, las hipótesis formuladas anteriormente se pueden escribir como:
# $H_0$:$μ1=μ2
# 
# $H_a$:$μ1>μ2
# 

# ### Paso 2: Seleccion de la prueba adecuada

# Esta es una prueba de una cola sobre dos proporciones de población de dos poblaciones independientes. **Con base en esta información, Se hara uso de la "Prueba Z" para 2 muestras**.

# ### Paso 3: Decidir el nivel de significancia

# Como se indica en el enunciado del problema, seleccionamos α = 0.05.

# ### Paso 4: recopilar y preparar datos

# In[38]:


# calculate the number of converted users in the treatment group
new_converted = df[df['group'] == 'treatment']['converted'].value_counts()['yes']
# calculate the number of converted users in the control group
old_converted = df[df['group'] == 'control']['converted'].value_counts()['yes'] # complete your code here
print('El numero de usuarios convertidos para la nueva y vieja pagina son {0} y {1} respectivamente'.format(new_converted, old_converted))

n_control = df.group.value_counts()['control'] # total number of users in the control group
n_treatment = df.group.value_counts()['treatment'] # total number of users in the treatment group
print('El numero de usuarios servidos para la nueva y vieja pagina son {0} y {1} respectivamente.'.format(n_control, n_treatment ))


# ### Paso 5: Calcular el valor p

# In[39]:


# complete the code to import the required function
from statsmodels.stats.proportion import proportions_ztest   

# write the code to calculate the p-value
test_stat, p_value = proportions_ztest([new_converted, old_converted] , [n_treatment, n_control], alternative ='larger')  



print('El valor P es:', p_value)


# ### Paso 6: Compare el valor p con $\alpha$

# In[40]:


# print the conclusion based on p-value
if p_value < 0.05:
    print(f'Como el valor P {p_value} es menor que la significancia, se rechaza la hipotesis nula.')
else:
    print(f'Como el valor P  {p_value} es mayor que la significancia, No se rechaza la hipotesis nula.')


# ### Paso 7: Extraer inferencia

# 
# Existe evidencia estadistica para rechazar la hipotesis nula y aceptar que la tasa de conversion o suscripcion de la pagina nueva es mayor la la tasa de conversion a la pagina antigua.
# 

# ## 3. ¿El estado convertido depende del idioma preferido?

# ### Realizar análisis visual

# In[41]:


# complete the code to visually plot the dependency between conversion status and preferred langauge
pd.crosstab(df['converted'],df['language_preferred'],normalize='index').plot(kind="bar", figsize=(6,8), stacked=True)
plt.legend()
plt.show()


# ### Paso 1: Definir las hipótesis nula y alternativa

# $H_0:$ la conversion a ser suscriptor no depende del idioma preferido.
# 
# 
# $H_a:$ la conversion a ser suscriptor depende del idioma preferido.
# 
# 

# ### Paso 2: Seleccione la prueba adecuada

# Este es un problema de la prueba de independencia, que concierne a dos variables categóricas: estatus convertido e idioma preferido. **Con base en esta información, se debe usar la prueba Chi cuadrada de independencia.**

# ### Paso 3: Decidir el nivel de significación

# Como se indica en el enunciado del problema, seleccionamos α = 0.05.

# ### Paso 4: recopilar y preparar datos

# In[42]:


# complete the code to create a contingency table showing the distribution of the two categorical variables
contingency_table = pd.crosstab(df['converted'], df['language_preferred'])  

contingency_table


# ### Paso 5: Calcular el valor p

# In[43]:


# complete the code to import the required function
from scipy.stats import chi2_contingency  

# write the code to calculate the p-value
chi2, p_value, dof, exp_freq = chi2_contingency(contingency_table)  

print('The p-value is', p_value)


# ### Paso 6: Compare el valor p con $\alpha$

# In[44]:


# print the conclusion based on p-value
if p_value < 0.05:
    print(f'Como el valor P {p_value} es menor que la significancia, se rechaza la hipotesis nula.')
else:
    print(f'As the p-value {p_value} es mayor que la significancia, No se rechaza la hipotesis nula.')


# ### Paso 7: Extraer inferencia

# Dado que el valor p es mayor que el nivel de significancia del 5%, existe suficiente evidencia estadistica para decir que el factor de conversion a ser sucriptor no depende del usuario.
# 

# ## 4. ¿El tiempo dedicado a la nueva página es el mismo para los usuarios de diferentes idiomas?

# ### Realizar análisis visual

# In[45]:


# create a new DataFrame for users who got served the new page
df_new = df[df['landing_page'] == 'new']


# In[46]:


# complete the code to visually plot the time spent on the new page for different language users
plt.figure(figsize=(8,8))
sns.boxplot(x = 'language_preferred', y = 'time_spent_on_the_page', showmeans = True, data = df_new)
plt.show()


# In[47]:


# complete the code to calculate the mean time spent on the new page for different language users
df_new.groupby(['language_preferred'])['time_spent_on_the_page'].mean()


# ### Paso 1: Definir las hipótesis nula y alternativa

# $H_0:$ Los tiempos medios de uso de las paginas son iguales para los 3 idiomas.
# 
# $H_a:$ Los tiempos medios de uso de las paginas no son iguales para los 3 idiomas.
# 
# 

# ### Paso 2: Seleccione la prueba adecuada

# Este es un problema relacionado con tres medias de población. **Con base en esta información, podemos usar la Prueba Anova unidireccional validando normalidad y varianza.**

# ### a. Prueba de Shapiro-Wilk

# hipótesis nula
# 
# $H
# 0
# :
#  $ El tiempo de uso en la nueva pagina tiene distribución normal
# 
# hipótesis alternativa
# 
# $H
# a
# :
#  $ El tiempo de uso en la nueva pagina tiene distribución normal

# In[48]:


#Import library
from scipy.stats import shapiro

# find the p-value
w, p_value = shapiro(df_new['time_spent_on_the_page']) 
print('The p-value is', p_value)


# No se rechaza hiptesis nula debido a que es mayor al nivel de significacia y se concluye con evidencia estadistica que el tiempo de uso en la nueva pagina tiene distribución normal.

# ### b. Prueba de Levene

# hipótesis nula
# 
# $H
# 0
# :$ Las varianzas poblacionales son iguales
# 
# hipótesis alternativa
# 
# $H
# a
# :$ Al menos una varianza es diferente del resto

# In[49]:


from scipy.stats import levene
statistic, p_value = levene( df_new[df_new['language_preferred']=="English"]['time_spent_on_the_page'], 
                             df_new[df_new['language_preferred']=="French"]['time_spent_on_the_page'], 
                             df_new[df_new['language_preferred']=="Spanish"]['time_spent_on_the_page'])
# find the p-value
print('The p-value is', p_value)


# El valor P es mayor al nivel de significancia no podemos rechazar la hipotesis nula, existe evidencia estadistica que las varianzas son iguales.

# ### Paso 3: Decidir el nivel de significancia

# Como se indica en el enunciado del problema, seleccionamos α = 0.05.

# ### Paso 4: recopilar y preparar datos

# In[50]:


# create a subsetted data frame of the time spent on the new page by English language users 
time_spent_English = df_new[df_new['language_preferred']=="English"]['time_spent_on_the_page']
# create subsetted data frames of the time spent on the new page by French and Spanish language users
time_spent_French = df_new[df_new['language_preferred']=='French']['time_spent_on_the_page']   
time_spent_Spanish = df_new[df_new['language_preferred']=='Spanish']['time_spent_on_the_page']


# ### Paso 5: Calcular el valor p

# In[51]:


# complete the code to import the required function
from scipy.stats import f_oneway 

# write the code to calculate the p-value
test_stat, p_value = f_oneway(time_spent_English, time_spent_French, time_spent_Spanish)   #complete the code by filling appropriate parameters in the blanks

print('The p-value is', p_value)


# ### Paso 6: Compare el valor p con $\alpha$

# In[52]:


# print the conclusion based on p-value
if p_value < 0.05:
    print(f'Dado que el Valor P {p_value} es menor que el nivel de significancia, se rechaza la hipotesis nula.')
else:
    print(f'Dado que el Valor P {p_value} es mayor que el nivel de significancia, se no rechaza la hipotesis nula.')


# ### Paso 7: Extraer inferencia

# Dado que el valor p de la prueba es mayor al nivel de significancia, existe evidencia estadistica para decir que los tiempos medios tienen mayor diferencia relacionados a el idioma.
# 
# 

# ## Conclusión y recomendaciones comerciales

# #### 1. Tiempo medio de interacción con la nueva página
# Conclusión:
# Dado que el valor p es 0.000139, mucho menor que el nivel de significancia de 0.05, se rechaza la hipótesis nula. Esto indica que el tiempo medio de interacción con la nueva página es significativamente mayor al tiempo medio de interacción con la página antigua.
# 
# $Recomendación$:
# Se debe implementar la nueva página de manera definitiva, ya que demuestra una mejora significativa en el tiempo de interacción de los usuarios, lo cual puede indicar un mayor compromiso y potencialmente mejores resultados comerciales.
# 
# #### 2. Tasa de conversión de la página nueva
# Conclusión:
# El valor p de 0.008026 es menor que el nivel de significancia de 0.05, lo que permite rechazar la hipótesis nula. Esto sugiere que la tasa de conversión de la nueva página es significativamente mayor que la de la página anterior.
# 
# $Recomendación$:
# Además de implementar la nueva página, se debe investigar más a fondo qué elementos específicos de la nueva página están contribuyendo al aumento en la tasa de conversión. Esto permitirá optimizar aún más las futuras versiones del sitio.
# 
# #### 3. Dependencia entre la conversión y el idioma preferido
# Conclusión:
# El valor p de 0.212988 es mayor que el nivel de significancia de 0.05, por lo que no se rechaza la hipótesis nula. Esto sugiere que la conversión no depende significativamente del idioma preferido.
# 
# $Recomendación$:
# Aunque la conversión no parece depender del idioma, es importante seguir monitoreando esta variable para detectar posibles cambios en el comportamiento de los usuarios multilingües. Además, se puede explorar si otros factores, como el contenido específico del idioma, pueden influir en la conversión.
# 
# #### 4. Tiempo dedicado a la nueva página según el idioma
# Conclusión:
# El valor p de 0.432041 es mayor que el nivel de significancia de 0.05, lo que indica que no hay diferencias significativas en el tiempo medio de uso de la página nueva entre los diferentes idiomas.
# 
# $Recomendación$:
# Dado que no hay diferencias significativas en el tiempo de uso entre los diferentes idiomas, se puede concluir que la nueva página es igualmente efectiva para usuarios de todos los idiomas. Sin embargo, es recomendable realizar análisis adicionales para explorar si hay otros factores que puedan influir en el tiempo de uso y mejorar la experiencia del usuario en función del idioma.
# 
# #### Resumen de Recomendaciones Generales:
# Implementación definitiva de la nueva página debido a los beneficios observados en el tiempo de interacción y la tasa de conversión.
# Investigación adicional sobre los elementos de la nueva página que contribuyen al aumento en la tasa de conversión.
# Monitoreo continuo de las tasas de conversión por idioma y análisis de otros factores que puedan influir en estas métricas.
# Análisis adicional para optimizar la experiencia del usuario según el idioma, aunque actualmente no se observen diferencias significativas.
# Estas conclusiones y recomendaciones están basadas en un análisis estadístico robusto y deberían ayudar a guiar futuras decisiones estratégicas para mejorar la interacción y conversión en el sitio web.

# ___
