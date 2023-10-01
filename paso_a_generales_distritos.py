# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:39:57 2023

@author: jfgonzalez
"""

from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import os


ARCHIVO = 'C:/Users/jfgonzalez/Downloads/2023-PROVISORIOS_PASO/paso_a_generales/ResultadosPorCircuito_ba_p_2019.csv'
ba_p = pd.read_csv(ARCHIVO)
ARCHIVO = 'C:/Users/jfgonzalez/Downloads/2023-PROVISORIOS_PASO/paso_a_generales/ResultadosPorCircuito_ba_g_2019.csv'
ba_g = pd.read_csv(ARCHIVO)

# %%

def pivoteo_y_paso_a_porcentaje(data_cap_presi):
    mesas_piv = data_cap_presi.pivot_table(values='votos',
                                            aggfunc=np.sum,
                                            columns='nom_agrupacion',
                                            index='cod_circuito')
    
    # ordeno las listas para que queden los partidos con más votos arriba
    listas_ordenadas = mesas_piv.sum().sort_values()
    listas_ordenadas.index
    mesas_piv= mesas_piv[listas_ordenadas.index]
    
    # saco mesas con pocos votos 
    # mesas_sub50 = mesas_piv[mesas_piv.sum(axis=1) <= 50]
    mesas_piv = mesas_piv[mesas_piv.sum(axis=1) > 50]
    
    # mesas_piv.drop('No positivo', inplace=True , axis=1)
    
    mesas_piv_porc = pd.DataFrame(columns=mesas_piv.columns, index=mesas_piv.index)
    for i in range(len(mesas_piv.index)):
        for j in range(len(mesas_piv.columns)):
            mesas_piv_porc.iloc[i, j] = (mesas_piv.iloc[i,j] / mesas_piv.iloc[i,:].sum())*100
    return mesas_piv_porc

# mesas_piv = pivoteo_y_paso_a_porcentaje(caba_p)

#%%


def elbow_clusters(mesas_piv, max_range=20, GRUPOS=10):
    Nc = range(1, max_range)
    
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    score = [kmeans[i].fit(mesas_piv).score(mesas_piv) for i in range(len(kmeans))]
    
    plt.plot([GRUPOS, GRUPOS], [score[2], score[-1]])
    plt.plot(Nc, score)
    plt.xlabel('Number of Clusters')
    plt.xticks(Nc)
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()

# elbow_clusters(mesas_piv, max_range=16, GRUPOS=12)

# %%

# calculo n=GRUPOS vectores representativos
def armo_clusters(mesas_piv, GRUPOS=10, grafico=True):
    kmeans = KMeans(n_init=20, max_iter=1000, n_clusters=GRUPOS).fit(mesas_piv)
    centroids = kmeans.cluster_centers_
    df_cent = pd.DataFrame(centroids, columns=mesas_piv.columns)
    
    if grafico:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
        
        ax.violinplot(df_cent, widths=1, vert=False, showextrema=False)
        ax.boxplot(df_cent, vert=False, sym='.', showcaps=False, showfliers=True)
        
        ax.set_yticks([y + 1 for y in range(len(df_cent.columns))],
                          labels=df_cent.columns)
    return kmeans

# kmeans = armo_clusters(mesas_piv, GRUPOS=12, grafico=True)

# %%

def asigno_grupos_y_etquetas(kmeans, mesas_piv):
    # armo grupos por vector representativo
    mesas_piv['grupo'] = kmeans
    
    #saco listado de mesas del indice para construir n_grupos
    mesas_piv['distrito'] = mesas_piv.index
    
    # n_grupos es el listado de mesas por grupo
    # listo las mesas
    n_grupos = pd.pivot_table(mesas_piv, index=['grupo'], values=['distrito'],
                              aggfunc=lambda x: ', '.join(x.astype(str)))
    # cuento las mesas
    n_grupos['cantidad_distr'] = pd.pivot_table(mesas_piv, index=['grupo'],
                                                values=['distrito'],aggfunc= 'count')
    
    # lo que sigue es una vuelta para ordenar el dataset y que queden
    # los grupos con más elementos arriba para ordenar después los graficos
    # de mayor a menor frecuencia (es probable que haya formas más eficientes)
    a= n_grupos.cantidad_distr.to_dict()
    # mesas_piv.rename(columns=nombre_lista, inplace=True)
    mesas_piv['orden'] = mesas_piv['grupo'].replace(a)
    n_grupos.sort_values('cantidad_distr', ascending=False, inplace=True)
    n_grupos['rank']= n_grupos.cantidad_distr.rank(ascending=False, method='first')-1
    mesas_piv.sort_values('orden', ascending=False,inplace=True, )
    a= n_grupos['rank'].to_dict()
    mesas_piv['grupo'] = mesas_piv['grupo'].replace(a).astype('int')
    n_grupos.index = n_grupos['rank']
    return n_grupos, mesas_piv

# n_grupos, mesas_piv = asigno_grupos_y_etquetas(kmeans, mesas_piv)

# %%


def grafico_grupos(mesas_piv, n_grupos, filas, columnas, figsize=(16,12), save=False, comentario=''):
    colores = []
    for agrup in mesas_piv.columns[:-3]:
        if agrup =='JUNTOS POR EL CAMBIO':
            colores.append('gold')
        elif agrup =='FRENTE DE TODOS':
            colores.append('skyblue')
        elif agrup =='UNITE POR LA LIBERTAD Y LA DIGNIDAD':
            colores.append('slateblue')
        elif agrup =='FRENTE DE IZQUIERDA Y DE TRABAJADORES - UNIDAD':
            colores.append('red')
        elif agrup == 'CONSENSO FEDERAL':
            colores.append('violet')
        else:
            colores.append('gray')
    
    
    fig, axs = plt.subplots(nrows=filas, ncols=columnas, figsize=figsize,
                            sharey=True, sharex=True)
    mesas_piv.grupo.unique()
    axs[0,0].set_xlim(0,100)
    for i, grup in enumerate(mesas_piv.grupo.unique()):
        titulo = (f'{n_grupos.loc[grup, "cantidad_mesas"]:,} distritos {comentario}')
        fila = int(i//columnas)
        columna = int(i - ((i//columnas)*columnas))
        df = mesas_piv[mesas_piv.grupo == grup].iloc[:,:-3].astype(float)
        
        v = axs[fila, columna].violinplot(df, vert=False, showextrema=False,
                                      widths=1)
        # asigno colores a cada partido
        for k, pc in enumerate(v['bodies']):
            pc.set_facecolor(colores[k])
        axs[fila, columna].boxplot(df,  vert=False, sym='.', showcaps=False,
                                   showfliers=True)
        axs[fila, columna].set_yticks([y + 1 for y in range(len(df.columns))],
                          labels=df.columns)
        
        axs[fila, columna].set_title("\n".join(wrap(titulo, 50)))
    
    
    fig.suptitle('Perfiles distritos, votos a presidente por agrupación - 2019', fontsize=16)
    fig.tight_layout(pad=2)
    if save:
        fig.savefig(f'grafico_dsitribuciones_{comentario}_grupos.png', dpi=90)
    return None


# grafico_grupos(mesas_piv, n_grupos, filas=4, columnas=3, figsize=(16,24))


# %%


dist_piv_p = pivoteo_y_paso_a_porcentaje(ba_p)
dist_piv_g = pivoteo_y_paso_a_porcentaje(ba_g)

kmeans_p = armo_clusters(dist_piv_p, GRUPOS=18, grafico=True)
kmeans_g = armo_clusters(dist_piv_g, GRUPOS=18, grafico=True)


dist_piv_g_rec = dist_piv_g[dist_piv_g.index.isin(dist_piv_p.index)]
dist_piv_p_rec = dist_piv_p[dist_piv_p.index.isin(dist_piv_g.index)]
k_lab = kmeans_p.labels_[dist_piv_p.index.isin(dist_piv_g.index)]


dist_piv_g_rec['otros'] = (dist_piv_g_rec['impugnados'] + dist_piv_g_rec['recurridos']
                          + dist_piv_g_rec['nulos'] # + dist_piv_g_rec['comando'] 
                           + dist_piv_g_rec['blancos'])
dist_piv_g_rec = dist_piv_g_rec.drop(['impugnados', 'recurridos', # 'comando',
                                      'nulos', 'blancos'], axis=1)

dist_piv_p_rec['otros'] = (dist_piv_p_rec['impugnados'] + dist_piv_p_rec['recurridos'] +
                           dist_piv_p_rec['MOVIMIENTO DE ACCION VECINAL'] 
                           + dist_piv_p_rec['PARTIDO AUTONOMISTA'] # + dist_piv_p_rec['comando']
                           + dist_piv_p_rec['FRENTE PATRIOTA'] + dist_piv_p_rec['nulos'] + 
                           dist_piv_p_rec['MOVIMIENTO AL SOCIALISMO'] + dist_piv_p_rec['blancos'])
dist_piv_p_rec = dist_piv_p_rec.drop(['impugnados', 'recurridos', 
                                      'MOVIMIENTO DE ACCION VECINAL',# 'comando',
                                      'PARTIDO AUTONOMISTA', 'FRENTE PATRIOTA', 
                                      'nulos','MOVIMIENTO AL SOCIALISMO', 'blancos'], axis=1)




n_grupos_p, dist_piv_p_rec = asigno_grupos_y_etquetas(k_lab, dist_piv_p_rec)
n_grupos_g, dist_piv_g_rec = asigno_grupos_y_etquetas(k_lab, dist_piv_g_rec)

grafico_grupos(dist_piv_p, n_grupos_p, filas=6, columnas=3, figsize=(16,24), comentario='PASO', save=True)
grafico_grupos(dist_piv_g, n_grupos_g, filas=6, columnas=3, figsize=(16,24), comentario='General', save=True)

# %%
from sklearn.decomposition import PCA

#%%

dist_piv_p = pivoteo_y_paso_a_porcentaje(ba_p)
dist_piv_g = pivoteo_y_paso_a_porcentaje(ba_g)


kmeans_p = armo_clusters(dist_piv_p, GRUPOS=12, grafico=True)
kmeans_g = armo_clusters(dist_piv_g, GRUPOS=12, grafico=True)

dist_piv_g_rec = dist_piv_g[dist_piv_g.index.isin(dist_piv_p.index)]
dist_piv_p_rec = dist_piv_p[dist_piv_p.index.isin(dist_piv_g.index)]
k_lab = kmeans_p.labels_[dist_piv_p.index.isin(dist_piv_g_rec.index)]


dist_piv_g_rec['otros'] = (dist_piv_g_rec['impugnados'] + dist_piv_g_rec['recurridos'] +
                           dist_piv_g_rec['nulos'] + dist_piv_g_rec['comando'] +
                           dist_piv_g_rec['blancos'])
dist_piv_g_rec = dist_piv_g_rec.drop(['impugnados', 'recurridos', 'comando',
                                      'nulos', 'blancos'], axis=1)

dist_piv_p_rec['otros'] = (dist_piv_p_rec['impugnados'] + dist_piv_p_rec['recurridos'] +
                           dist_piv_p_rec['MOVIMIENTO DE ACCION VECINAL'] 
                           + dist_piv_p_rec['PARTIDO AUTONOMISTA'] + dist_piv_p_rec['comando']
                           + dist_piv_p_rec['FRENTE PATRIOTA'] + dist_piv_p_rec['nulos'] + 
                           dist_piv_p_rec['MOVIMIENTO AL SOCIALISMO'] + dist_piv_p_rec['blancos'])
dist_piv_p_rec = dist_piv_p_rec.drop(['impugnados', 'recurridos', 
                                      'MOVIMIENTO DE ACCION VECINAL','comando',
                                      'PARTIDO AUTONOMISTA', 'FRENTE PATRIOTA', 
                                      'nulos','MOVIMIENTO AL SOCIALISMO', 'blancos'], axis=1)



#%%


pca = PCA(n_components=2, random_state=22)
pca.fit(dist_piv_p_rec)
x = pca.transform(dist_piv_p_rec)

n_grupos_p, dist_piv_p_rec = asigno_grupos_y_etquetas(k_lab, dist_piv_p_rec)
n_grupos_g, dist_piv_g_rec = asigno_grupos_y_etquetas(k_lab, dist_piv_g_rec)

pd.DataFrame(x).plot.scatter(x=0, y=1)
pca.explained_variance_ratio_
#%%

import seaborn as sns

df_x = pd.DataFrame(x, columns=['Comp1', 'Comp2'], index=dist_piv_p_rec.index)

df_x['grupo'] = dist_piv_p_rec['grupo']


pd.DataFrame(x).plot.scatter(x=0, y=1, )

sns.scatterplot(x=df_x['Comp1'], y=df_x['Comp2'], hue=df_x['grupo'] )


#%%

'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Crear el modelo secuencial
model = Sequential()

# Agregar una capa de entrada con 20 dimensiones
model.add(Dense(64, input_dim=20, activation='relu'))

# Agregar una capa oculta (puedes ajustar el número de neuronas según sea necesario)
model.add(Dense(32, activation='relu'))

# Agregar la capa de salida con 10 dimensiones (correspondientes a a0, b0, ...)
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Resumen del modelo
model.summary()

# %%

# X_train es tu conjunto de datos de entrenamiento de entrada (vectores de 20 dimensiones)
# y_train es tu conjunto de datos de etiquetas de salida (vectores de 10 dimensiones)

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

'''

