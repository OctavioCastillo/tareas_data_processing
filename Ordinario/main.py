# main.py
from regresion_lineal import RegresionLineal
from red_bayesiana import RedBayesiana
#from red_neuronal import RedNeuronal
#from algoritmo_genetico import AlgoritmoGenetico

def mostrar_menu():
    print("\n====== MENÚ DE MODELOS DE PREDICCIÓN ======")
    print("1. Regresión Lineal")
    print("2. Red Bayesiana")
    print("3. Red Neuronal")
    print("4. Algoritmo Genético")
    print("0. Salir")
    opcion = input("Seleccione una opción: ")
    return opcion

def main():
    while True:
        opcion = mostrar_menu()

        if opcion == '1':
            modelo = RegresionLineal()
            modelo.ejecutar()
        elif opcion == '2':
            modelo = RedBayesiana()
            modelo.ejecutar()
        # elif opcion == '3':
        #     modelo = RedNeuronal()
        #     modelo.predecir()
        # elif opcion == '4':
        #     modelo = AlgoritmoGenetico()
        #     modelo.predecir()
        elif opcion == '0':
            print("Saliendo del programa. ¡Hasta luego!")
            break
        else:
            print("Opción no válida. Por favor, elija una opción del 0 al 4.")

if __name__ == "__main__":
    main()
