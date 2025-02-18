#adaptado de https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
#Obteniendo informacion
img = cv2.cvtColor(cv2.imread('au.png'), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("Original")
plt.show()
#Dividiendo en canales de color
r, g, b = img[:,:,0], img[:,:,1], img[:,:,2] #si usamos r g b como canales de color en vez de cv2.split el color se ve raro, revisar valores de las matrices
gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

blue,green,red = cv2.split(img)
fig = plt.figure() 
fig.add_subplot(131)
plt.title("canal azul")
plt.imshow(blue,cmap='gray')
fig.add_subplot(132)
plt.title("canal verde")
plt.imshow(green,cmap='gray')
fig.add_subplot(133)
plt.title("canal rojo")
plt.imshow(red,cmap='gray')
fig.add_subplot(332)
plt.title("imagen gris")
plt.imshow(gray,cmap='gray')
plt.show()
#normalizando
df_blue = blue/255
df_green = green/255
df_red = red/255
df_gray = gray/255
#aplicando pca
pca_gray = PCA(n_components=100)
pca_gray.fit(df_gray)
trans_pca_gray = pca_gray.transform(df_gray)
pca_b = PCA(n_components=100)
pca_b.fit(df_blue)
trans_pca_b = pca_b.transform(df_blue)
pca_g = PCA(n_components=100)
pca_g.fit(df_green)
trans_pca_g = pca_g.transform(df_green)
pca_r = PCA(n_components=100)
pca_r.fit(df_red)
trans_pca_r = pca_r.transform(df_red)
#Explicacion de cada canal
print(f"Imagen gris : {sum(pca_gray.explained_variance_ratio_)}")
print(f"Canal azul  : {sum(pca_b.explained_variance_ratio_)}")
print(f"canal verde : {sum(pca_g.explained_variance_ratio_)}")
print(f"canal rojo  : {sum(pca_r.explained_variance_ratio_)}")
#regresando los valores de PCA a matriz de imagenes
b_arr = pca_b.inverse_transform(trans_pca_b)
g_arr = pca_g.inverse_transform(trans_pca_g)
r_arr = pca_r.inverse_transform(trans_pca_r)
#PCA en cada canal de color
fig = plt.figure()
fig.add_subplot(131)
plt.title("canal azul")
plt.imshow(blue,cmap='gray')
fig.add_subplot(132)
plt.title("canal verde")
plt.imshow(green,cmap='gray')
fig.add_subplot(133)
plt.title("canal rojo")
plt.imshow(red,cmap='gray')
fig.add_subplot(231)
plt.title("canal azul reducido")
plt.imshow(b_arr,cmap='gray')
fig.add_subplot(232)
plt.title("canal verde reducido")
plt.imshow(g_arr,cmap='gray')
fig.add_subplot(233)
plt.title("canal rojo reducido")
plt.imshow(r_arr,cmap='gray')
plt.show()
#explicacion de cada componente
exp_var_pca = pca_gray.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Explicacion de cada componente')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Explicacion acumulada')
plt.ylabel('Porcentaje de explicacion')
plt.xlabel('Componente principal')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
#creacion de las imagenes
img_reduced = (cv2.merge((b_arr, g_arr, r_arr)))
img_reduced_gris = pca_gray.inverse_transform(trans_pca_gray)
#Comparacion imagenes a color
fig = plt.figure() 
fig.add_subplot(121)
plt.title("Imagen Original")
plt.imshow(img)
fig.add_subplot(122)
plt.title("Imagen Comprimida")
plt.imshow(img_reduced)
plt.show()
#Comparacion imagenes a escala de grises
fig = plt.figure() 
fig.add_subplot(121)
plt.title("Imagen Original Gris")
plt.imshow(gray, cmap='gray')
fig.add_subplot(122)
plt.title("Imagen Comprimida Gris")
plt.imshow(img_reduced_gris, cmap='gray')
plt.show()

