import random, sys, math

# Nota: en lugar de matrices se usan listas de listas

# Generauna matriz de distancias de nCiudaddes x nCiudades
def matrizDistancias(nCiud, distanciaMaxima):
    matriz = [[0 for i in range(nCiud)] for j in range(nCiud)]

    for i in range(nCiud):
        for j in range(i):
            matriz[i][j] = distanciaMaxima*random.random()
            matriz[j][i] = matriz[i][j]

    return matriz

# Elige un paso de una hormiga, teniendo encuenta las distancias
# y las feromonas y descartando las cuidades ya visitadas
def eligeCiudad(dists, ferom, visitadas):
    # Se calcula la tabla de pesos de cada ciudad
    listaPesos = []
    disponibles = []
    actual = visitadas[-1]

    # Influencia de cada valor
    alfa = 1.0
    beta = 0.5

    # El parámetro beta ( peso de distancias) es 0.5, alfa = 1.0
    for i in range(len(dists)):
        if i not in visitadas:
            fer = math.pow((1.0 + ferom[actual][i]), alfa)
            peso = math.pow(1.0/dists[actual][i], beta) * fer
            disponibles.append(i)
            listaPesos.append(peso)

    # Se elige aleatoriamente una de las ciudades disponibles,
    # teniendo en cuenta su peso relativo
    valor = random.random() * sum(listaPesos)
    acumulado = 0.0
    i = -1
    while valor > acumulado:
        i += 1
        acumulado += listaPesos[i]

    return disponibles[i]

# Genera una "hormiga", que elegirá un camino teniendo en cuenta
# las distancias y los rastros de feromonas. Devuelve una tupla
# con el camino y su longitud
def eligeCamino(distancias, feromonas):
    # La ciudad inicial siempre es la 0
    camino = [0]
    longCamino = 0

    # Elegir cada paso según la distancia y las feromonas
    while len(camino) < len(distancias):
        ciudad = eligeCiudad(distancias, feromonas, camino)
        longCamino += distancias[camino[-1]][ciudad]
        camino.append(ciudad)

    # Para terminar hay que volver a la ciudad de origen (0)
    longCamino += distancias[camino[-1]][0]
    camino.append(0)

    # Para terminar hey que volver a la ciudad de origen (0)
    longCamino += distancias[camino[-1]][0]
    camino.append(0)

    return (camino, longCamino)

# Actualiza la matriz de feromonas siguiendo el camino recibido
def rastroFeromonas(feromonas, camino, dosis):
    for i in range(len(camino) - 1):
        feromonas[camino[i]][camino[i+1]] += dosis

# Evapora todas las feromonas multiplicándolas por una constante
# = 0.9 (en otra palabras, el coeficiente de evaporación es 0.1)
def evaporaFeromanas(feromonas):
    for lista in feromonas:
        for i in range(len(lista)):
            lista[i] *= 0.9

# Resuelve el problema dle viajante de comercio mediante el 
# algoritmo de la colonia de hormigas. Recibe una matriz de 
# distancias y devuelve una tupla con el mejor camino que ha
# obtenido (lista de índices) y su longitud
def hormigas(distancias, iteraciones, distMedia):
    # Primero se crea una matriz de feromonas vacía
    n = len(distancias)
    feromonas = [[0 for i in range(n)] for j in range(n)]

    # El mejor camino y longitud (inicialmente "infinita")
    mejorCamino = []
    longMejorCamino = float('inf')

    # En cada iteración se genera una hormiga, que elige un camino,
    # y si es mejor que el mejor que teníamos, deja su rastro de 
    # feromonas (mayor cuanto más corto sea el camino)
    for  iter in range(iteraciones):
        (camino,longCamino) = eligeCamino(distancias, feromonas)

        if longCamino <= longMejorCamino:
            mejorCamino = camino
            longMejorCamino = longCamino

        rastroFeromonas(feromonas, camino, distMedia/longCamino)

        # En cualquier caso, las feromonas se van explorando
        evaporaFeromanas(feromonas)

    # Se devuelve el mejor camino que se haya encontrado
    return (mejorCamino, longMejorCamino)

# Generación de una matriz de prueba
numCiudades = 10
distanciaMaxima = 10
ciudades = matrizDistancias(numCiudades, distanciaMaxima)

# Obtención del mejor camino
iteraciones = 1000
distMedia = numCiudades*distanciaMaxima/2
(camino, longCamino) = hormigas(ciudades, iteraciones, distMedia)
print("Camino:_", camino)
print("Longitud_del_camino:_", longCamino)


