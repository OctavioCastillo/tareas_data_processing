from random import random

# Función que se requiere minimizar
def funcion(x, y):
    sum1 = x**2 * (4-2.1*x**2 + x**4/3.0)
    sum2 = x*y
    sum3 = y**2 * (-4+4*y**2)
    return sum1 + sum2 + sum3

# Devuelve un número aleatorio dentro de un rango con 
# distribución uniforme (proporcionada por random)
def aleatorio(inf, sup):
    return random()*(sup-inf)+inf

# Clase que representa una partícula individual y que facilita
# las operaciones necesarias
class Particula:
    # Algunos atributos de clase (comunes a todas las partículas)
    # Parámetros para actualizar la velocina
    inercia = 1.4
    cognitiva = 2.0
    social = 2.0
    # Limites del espacio de soluciones
    infx = -2.0
    supx = 2.0
    infy = -1.0
    supy = 1.0
    # Factor de ajuste de la velocidad inicial
    ajusteV = 100.0

    # Crea una partícula dentro de los límites indicadas
    def __init__(self):
        self.x = aleatorio(Particula.infx, Particula.supx)
        self.y = aleatorio(Particula.infy, Particula.supy)
        self.vx = aleatorio(Particula.infx/Particula.ajusteV, Particula.supx/Particula.ajusteV)
        self.vy = aleatorio(Particula.infy/Particula.ajusteV, Particula.supy/Particula.ajusteV)
        self.xLoc = self.x
        self.yLoc = self.y
        self.valorLoc = funcion(self.x, self.y)

    # Actualiza la velocidad de la partícula
    def actualizaVelocidad(self, xGlob, yGlob):
        cogX = Particula.cognitiva*random()*(self.xLoc-self.x)
        socX = Particula.social*random()*(xGlob-self.x)
        self.vx = Particula.inercia*self.vx + cogX + socX
        cogY = Particula.cognitiva*random()*(self.yLoc-self.y)
        socY = Particula.social*random()*(yGlob-self.x)
        self.vy = Particula.inercia*self.vy + cogY + socY

    # Actualiza la posición de la partícula
    def actualizaPosicion(self):
        self.x = self.x + self.vx
        self.y = self.y + self.vy

        # Debe mantenerse dentro del espacio de soluciones
        self.x = max(self.x, Particula.infx)
        self.x = min(self.x, Particula.supx)
        self.y = max(self.y, Particula.infy)
        self.y = min(self.y, Particula.supy)

        # Si es inferior a lo mejor, la adopta como mejor 
        valor = funcion(self.x, self.y)
        if valor < self.valorLoc:
            self.xLoc = self.x
            self.yLoc = self.y
            self.valorLoc = valor

# Mueve un enjambre de partículas durante las interaciones indicadas
# Devuelve las coordenadas y el valor del mínimo obtenido
def enjambreParticulas(particulas, iteraciones, reduccionInercia):

    # Registra la mejor posición global y su valor
    mejorParticula = min(particulas, key=lambda p:p.valorLoc)
    xGlob = mejorParticula.xLoc
    yGlob = mejorParticula.yLoc
    valorGlob = mejorParticula.valorLoc

    # Bucle principal de simulación
    for iter in range(iteraciones):
        # Actualiza la velocidad y posición de cada partícula
        for p in particulas:
            p.actualizaVelocidad(xGlob, yGlob)
            p.actualizaPosicion()

        # Hasta que no se han movido todas las partículas no se
        # actualiza el mínimo global, para simular que todas se 
        # mueven a la vez
        mejorParticula = min(particulas, key=lambda p:p.valorLoc)
        if mejorParticula.valorLoc < valorGlob:
            xGlob = mejorParticula.xLoc
            yGlob = mejorParticula.yLoc
            valorGlob = mejorParticula.valorLoc

        # Finalmente se reduce la inercia de las partículas
        Particula.inercia *= reduccionInercia

    return (xGlob, yGlob, valorGlob)

# Parámetros del problema
nParticulas = 10
iteraciones = 100
redInercia = 0.9

#Genera un conjunto inicial de partículas
particulas = [Particula() for i in range(nParticulas)]

# Ejecuta el algoritmo del enjambre de partículas
print(enjambreParticulas(particulas, iteraciones, redInercia))