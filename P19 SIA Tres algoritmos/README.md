Comprensión de Conceptos

1. ¿Qué similitudes encuentras entre el sistema inmune biológico y el algoritmo de selección clonal implementado en el código SeleccionClonal.py?
Ambos seleccionan las mejores respuestas (anticuerpos o soluciones), las clonan y las mutan para mejorar su desempeño frente a un problema o antígeno.

2. ¿Qué representa la función rastrigin en el contexto del algoritmo? ¿Por qué es útil para evaluar soluciones?
Es una función de prueba matemática con muchos mínimos locales. Se usa para evaluar qué tan buena es una solución; valores más bajos indican mejor aptitud.

3. En el código SeleccionNegativa.py, ¿qué representan los detectores y cómo se relacionan con el concepto de "tolerancia al propio"?
Los detectores son puntos generados aleatoriamente que no deben parecerse a los datos normales. Esto simula la tolerancia al propio, es decir, el sistema no debe atacar células propias.

4. ¿Por qué es importante que los detectores no se parezcan a los datos normales? ¿Qué pasaría si sí se parecieran?
Porque si se parecen, detectarían erróneamente datos normales como anomalías, generando falsos positivos.

5. ¿Qué papel juega la mutación en el algoritmo de selección clonal? ¿Cómo se relaciona con la diversidad de soluciones?
La mutación introduce variaciones en los clones, lo que permite explorar nuevas áreas del espacio de búsqueda y evita la convergencia prematura.

6. ¿Qué representa la “población” en el algoritmo de RedesInmunitarias.ipynb y cómo se relaciona con el sistema inmune?
Representa posibles soluciones (vectores de pesos). Simula la variedad de anticuerpos que evolucionan para responder a distintos estímulos.

7. ¿Qué papel juega la red de vecinos (immune network) en el proceso de evolución de soluciones?
Permite que las soluciones influyan entre sí a través de relaciones de similitud, promoviendo la diversidad y guiando la evolución.

8. ¿Por qué se utiliza una función de aptitud basada en el error cuadrático medio (MSE)?
Porque mide con precisión la diferencia entre las predicciones y los valores reales. Es ideal para problemas de regresión.

9. ¿Qué significa que una solución tenga “mejor aptitud” en este contexto?
Que su error cuadrático medio (MSE) es menor, es decir, predice con mayor exactitud.

10. ¿Cómo se relaciona la mutación con la diversidad de anticuerpos en el sistema inmune natural?
En ambos casos, la mutación permite generar nuevas variantes para responder mejor a amenazas o encontrar soluciones más óptimas.




Análisis de código

11. En SeleccionClonal.py, identifica la parte del código que realiza la clonación de los mejores candidatos. ¿Qué función se encarga de esto?
La función clone_candidates(candidates, num_clones) realiza la clonación usando np.repeat().

12. ¿Qué función se encarga de introducir variaciones en los clones? ¿Cómo se implementa la mutación?
La función mutate_clones(clones, mutation_rate) aplica mutaciones sumando un valor aleatorio a elementos seleccionados por probabilidad.

13. En SeleccionNegativa.py, ¿cómo se asegura el algoritmo de que los detectores no se parezcan a los datos normales? ¿Qué función lo hace?
La función generate_detectors() usa np.allclose() con atol=0.5 para verificar que los nuevos detectores no estén cerca de datos normales.

14. ¿Cómo se calcula la distancia entre un detector y un punto de datos? ¿Qué métrica se utiliza?
Se utiliza la norma euclidiana con np.linalg.norm(detector - point).

15. ¿Qué parte del código se encarga de visualizar los resultados? ¿Qué información útil aporta esta visualización?
Desde plt.figure(figsize=(14, 6)) hasta plt.show() se crean dos subplots. Muestran datos normales, fraudulentos, detectores, anomalías y la frontera de decisión, permitiendo validar el rendimiento visual del algoritmo.

16. En las RedesInmunitarias.ipynb ¿Qué función genera la población inicial de soluciones? ¿Qué valores iniciales se usan?
La función generate_initial_population(pop_size, solution_size) genera soluciones con valores aleatorios entre -1 y 1.

17. ¿Qué parte del código se encarga de construir la red de vecinos? ¿Cómo se determina quién es vecino de quién?
La función create_immune_network() calcula distancias entre soluciones y selecciona los vecinos más cercanos usando np.argsort().

18. ¿Cómo se implementa la mutación en el código RedesInmunitarias.ipynb? ¿Qué parámetros controlan su intensidad?
La función update_network() suma pequeñas perturbaciones aleatorias. La intensidad está controlada por mutation_rate y el rango de np.random.uniform(-0.05, 0.05).

19. ¿Qué parte del código RedesInmunitarias.ipynb selecciona las mejores soluciones para la siguiente generación?
Dentro de immune_network_theory(), se combinan poblaciones y se seleccionan las mejores usando np.argsort(combined_fitness)[:pop_size].

20. ¿Cómo se visualiza la mejora del algoritmo RedesInmunitarias.ipynb a lo largo de las generaciones? ¿Qué representa la gráfica generada?
plt.plot(best_fitness_per_generation, ...)
grafica cómo disminuye el error (MSE) generación tras generación. Representa el progreso del algoritmo hacia soluciones más precisas.




Aplicaciones

21. ¿Cómo modificarías el algoritmo de selección clonal para resolver un problema de clasificación en lugar de optimización?
Cambiar la función de evaluación para que mida precisión o alguna métrica de clasificación, y usar un conjunto de datos etiquetado.

22. ¿Qué cambios harías en el algoritmo de selección negativa si los datos normales tuvieran más de dos características (dimensiones)?
Modificar el parámetro detector_size para igualarlo a la nueva dimensión y ajustar atol o el umbral de distancia.

23. ¿Cómo podrías ajustar el umbral de detección (threshold) en SeleccionNegativa.py para reducir los falsos positivos?
Incrementar el valor de threshold en la función detect_anomalies() para hacer la detección más estricta.

24. ¿Qué otras funciones objetivo podrías usar en lugar de rastrigin para probar el algoritmo de selección clonal?
Ackley, Sphere, Rosenbrock, Schwefel, o funciones personalizadas para casos reales.

25. ¿Cómo podrías usar el algoritmo de selección negativa para detectar anomalías en datos de sensores en tiempo real?
Entrenar detectores con datos históricos normales y luego comparar continuamente nuevos datos para detectar desviaciones.

26. ¿Cómo modificarías el algoritmo RedesInmunitarias.ipynb para predecir otra variable económica, como la inflación o el tipo de cambio?
Reemplazar y por la nueva variable objetivo y ajustar las entradas X con los indicadores relevantes para esa predicción.

27. En RedesInmunitarias.ipynb ¿Qué cambios harías si los datos tuvieran más características (por ejemplo, 20 indicadores en lugar de 5)?
Ajustar n_features, la dimensión de cada solución, y posiblemente aumentar el tamaño de la población para mantener diversidad.

28. ¿Cómo podrías adaptar el enfoque de Redes Inmunitarias para un problema de clasificación binaria (por ejemplo, detectar si una acción subirá o bajará)?
Modificar la función de aptitud para utilizar una métrica como entropía cruzada o precisión, y ajustar la salida del modelo para ser binaria.

29. ¿Qué otras funciones de aptitud podrías usar si el objetivo fuera minimizar el error absoluto en lugar del cuadrático con Redes Inmunitarias?
Usar MAE (Mean Absolute Error) en lugar de MSE en fitness_function().

30. ¿Cómo podrías usar el algoritmo RedesInmunitarias.ipynb en tiempo real, por ejemplo, para ajustar predicciones con datos que llegan continuamente?
Implementar aprendizaje incremental o actualización en línea, donde las nuevas muestras se incorporen y actualicen la población periódicamente.
