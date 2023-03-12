# -*- coding: utf-8 -*-
"""
# Pytorch: Tensores
"""

import torch
import numpy as np

#####################
### Pytorch: Tensores
#####################

### Que es un Tensor en Pytorch?
# Un tensor es similar a una matriz o un vector, pero se extiende a múltiples 
# dimensiones. Por ejemplo, un tensor 1D puede ser visto como un vector, mientras
# que un tensor 2D puede ser visto como una matriz. Un tensor 3D puede ser visto
# como un conjunto de matrices, y así sucesivamente.

### Declarando tensores
#En PyTorch, los tensores son objetos de la clase torch.Tensor. Un tensor puede 
#ser creado utilizando la función torch.tensor(), que toma una lista, una tupla 
#o un arreglo NumPy como argumento y devuelve un tensor PyTorch. Por ejemplo, 
#para crear un tensor 1D con los valores [1, 2, 3], podemos escribir:

tensor_1d = torch.tensor([1, 2, 3])

#Para crear un tensor 2D 
tensor_2d = torch.tensor([[1, 2], [3, 4]]) 

# Tensor 3d o mas,en este caso 2 elementos por dimensión y asi sucesivamente
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])




### Tensores a partir de arreglos/matrices de Numpy
# Numpy es esta libreria para trabajar matematicamente con arreglos/matrices
# podemos hacer una especie de Parse de Numpy matrices a Tensores

numpy_array = np.array([[1, 2], [3, 4]]) # Matriz de Numpy
tensor = torch.from_numpy(numpy_array)   # la hacemos Tensor




### Generar Tensores a partir de una base 
# Podemos generar nuevos tensores que conserven la estructura de uno existente

x_ones = torch.ones_like(tensor_3d) # Conserva las propiedades de tensor_3d
print(f"Ones Tensor: \n {x_ones} \n")

shape = (2,3,) # Dos dimensiones, 3 elementos
rand_tensor = torch.rand(shape) # todos conservan la forma
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)




### Procesamiento Paralelo de GPU
# Por defecto utilizamos los nucleos de la CPU pero si existe la disponibildad
# de utilizar la GPU, obtendremos la ventaja de sus multiples nucleos y capacidad
# de procesar en paralelo y no secuencial.
if torch.cuda.is_available():
  tensor = tensor.to('cuda')




### Forma de buscar en tensores
# Esto es parecido a matrices, aqui tienes un ejemplo

tensor = torch.ones(4, 4)             # 4 dimensiones con 4 elementos

tensor[:,1] = 0                       # Le decimos que la segunda columana sea 0s,
                                      # 2da por que se empieza en 0, no en 1
print('Seleccionar en Tensores\nFirst row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])


print(tensor)




### Concatenar Tensores
# Con torch.cat() podemos concatenar(unir) 2 o mas tensores su sintaxis es:

#torch.cat(seq, dim=0, *, out=None) -> Tensor

# seq: una secuencia de tensores que se van a concatenar.
# dim: la dimensión a lo largo de la cual se va a concatenar. Por defecto es 0.
# out (opcional): un tensor de destino donde se almacenará el resultado de la concatenación.

# Ejemplo:

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.tensor([[9, 10], [11, 12]])

d = torch.cat((a, b, c), dim=0)

print("\nImprimos una concatenacion de tensores",d)




### Operaciones con  y de Tensores
# Podemos sumar el total de un tensor con .sum()

x = torch.tensor([[1, 2], [3, 4]])
total_sum = x.sum()
print(total_sum)        # da el total de tensor en formato tensor(total)
print(total_sum.item()) # Con .item() lo hacemos un simple numero




### Modificaciones directamente en el Objeto "In place Operand_"
# Se suelen denotar por _
# ejemplo tensorIn.copy_(tensor)
# tensorIn.t_
tensorIn = torch.tensor([[3, 2], [5, 1]])

print("In place Operand, remplazamos en la variable",tensorIn)
tensorIn.add_(5)
print("le sumamos 5 a cada elemento",tensorIn)