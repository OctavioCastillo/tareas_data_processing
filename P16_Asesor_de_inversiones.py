from random import random, randint, sample
from collections import namedtuple

# Calcula el capital invertido por un indiciduo
def capitalInvertido(individuo):
    return sum(map(lambda x, y: x*y.precio, individuo, inversiones))

# Calcula el rendimineto obtenido por un individuo
def rendimiento(individuo):
    return sum(map(lambda x,y: x*y.precio*y.rendim, individuo, inversiones))

# Si un individuo gasta más capital disponible, se eliminan
# aleatoriamente inversiones hasta que se ajusta al capital
def ajustaCapital(individuo):
    ajustado = individuo[:]
    while capitalInvertido(ajustado)>capital:
        pos = randint(0,len(ajustado)- 1)
        if ajustado[pos] > 0:
            ajustado[pos] -= 1
    return ajustado

# Crea un individuo al azar, en este caso una selección de
# inversiones que no excedan el capital disponible
def creaIndividuo(inversiones, capital):
    individuo = [0]*len(inversiones)

    while capitalInvertido(individuo) < capital:
        eleccion = randint(0, len(inversiones)-1)
        individuo[eleccion] += 1

    return ajustaCapital(individuo)

# Crea un nuevo individuo cruzando otros dos 
def cruza(poblacion, posiciones):
    L = len(poblacion[0])

    hijo = poblacion[posiciones[0]][:]
    inicio = randint(0,L-1)
    fin = randint(inicio+1,L)
    hijo[inicio:fin] = poblacion[posiciones[1]][inicio:fin]

    return ajustaCapital(hijo)


# Aplica mutaciones a un individuo segun una tasa dada; garantiza
# que cumple las restricciones de capital e inversiones
def muta(individuo, tasaMutacion):
    mutado = []
    for i in range(len(individuo)):
        if random() > tasaMutacion:
            mutado.append(individuo[i])
        else:
            mutado.append(randint(0, inversiones[i].cantidad))

    return ajustaCapital(mutado)

# Hacer evolucionar el sistema durante un número de genereaciones
def evoluciona(poblacion, generaciones):

    # Orden la población incial por rendimiento produciodo
    poblacion.sort(key=lambda x:rendimiento(x))

    # Algunos valores útiles
    N = len(poblacion)
    tasaMutacion = 0.01

    # Genera una lista del tipo [0, 1, 1, 1, 2, 2, 3, 3...] para 
    # representar las probabilidades de reprosucirse de cada individuo

    reproduccion = [x for x in range(N) for y in range(x+1)]

    for i in range(generaciones):
        # Se generan N-1 nuevos individuos cruzando los existentes
        # (sin que se repitan los padres)
        padres = sample(reproduccion, 2)
        while padres[0] == padres[1]:
            padres = sample(reproduccion,2)

        hijos = [cruza(poblacion, padres) for x in range(N-1)]

        # Se aplican mutaciones con una cierta probabilidad
        hijos = [muta(x, tasaMutacion) for x in hijos]

        # Se añade el mejor individuo de la población anterior
        # (elitismo)
        hijos.append(poblacion[-1])
        poblacion = hijos

        # Se ordenan os individuos por rendimiento
        poblacion.sort(key=lambda x:rendimiento(x))

    # Devuelve el mejor individuo encontrado
    return poblacion[-1]

# Declara una tupa con nombres para representar cada inversión
Inversion = namedtuple('Inversion', ['precio', 'cantidad', 'rendim'])


numInver = 100
maxPrecio = 1000
maxCant = 10
maxRend = 0.2

# Gerenera una lista de tuplas Inversión
inversiones = [Inversion(random()*maxPrecio, randint(1, maxCant), random()*maxRend) for i in range(numInver)]

print(inversiones)

capital = 50000
individuos = 20
generaciones = 1000

poblacion = [creaIndividuo(inversiones, capital)
             for i in range(individuos)]

# Nota: para simplificar el programa se accede a inversiones y 
# capital de forma global (solo se leen, no se modifican)

mejor = evoluciona(poblacion, generaciones)
print(mejor, capitalInvertido(mejor), rendimiento(mejor))

import matplotlib.pyplot as plt

# Índices de las inversiones (0, 1, 2, ..., numInver-1)
indices = list(range(len(mejor)))
cantidades = mejor  # número de bonos seleccionados por tipo de inversión

# Para mostrar sólo las inversiones efectivamente usadas
indices_usados = [i for i, c in enumerate(cantidades) if c > 0]
cantidades_usadas = [cantidades[i] for i in indices_usados]
rendimientos_usados = [inversiones[i].rendim for i in indices_usados]
precios_usados = [inversiones[i].precio for i in indices_usados]

# Gráfico de barras: cantidad de bonos por tipo
plt.figure(figsize=(12, 6))
plt.bar(indices_usados, cantidades_usadas, color='skyblue')
plt.xlabel('Índice de Inversión')
plt.ylabel('Cantidad seleccionada')
plt.title('Distribución de Bonos Seleccionados por el Mejor Individuo')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
