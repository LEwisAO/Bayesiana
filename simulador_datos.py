import numpy as np
import pandas as pd

# Configuración de la semilla para reproducibilidad
np.random.seed(50)

# Función para generar datos ficticios con dependencia de IMC en NivelActividad y Edad
def generar_datos_dependientes_imc(num_datos):
    generos = ['Hombre', 'Mujer']
    niveles_actividad = ['Bajo', 'Moderado', 'Alto']
    niveles_estres = ['Bajo', 'Medio', 'Alto']
    consumos_azucar = ['Bajo', 'Moderado', 'Alto']
    historiales_familiares = ['Sí', 'No']

    datos = {
        'Género': np.random.choice(generos, num_datos),
        'Edad': np.random.randint(20, 60, num_datos),
        'NivelActividad': np.random.choice(niveles_actividad, num_datos),
        'IMC': []
    }

    # Dependencia de IMC en NivelActividad y Edad
    for i in range(num_datos):
        if datos['NivelActividad'][i] == 'Bajo':
            datos['IMC'].append(np.random.uniform(25, 30))
        elif datos['NivelActividad'][i] == 'Moderado':
            datos['IMC'].append(np.random.uniform(20, 25))
        else:
            datos['IMC'].append(np.random.uniform(18, 23))

        # Añadir influencia de la Edad
        datos['IMC'][i] += 0.1 * datos['Edad'][i]

    datos['IMC'] = np.clip(datos['IMC'], 18, 35)  # Asegurar que el IMC esté en un rango razonable

    # Resto de las variables
    datos['Glucosa'] = np.random.randint(80, 160, num_datos)
    datos['Insulina'] = np.random.uniform(5, 20, num_datos)
    datos['NivelEstrés'] = np.random.choice(niveles_estres, num_datos)
    datos['ConsumoAzúcares'] = np.random.choice(consumos_azucar, num_datos)
    datos['HistorialFamiliar'] = np.random.choice(historiales_familiares, num_datos)
    
    # Dependencia de RiesgoDiabetes en Glucosa
    datos['RiesgoDiabetes'] = np.where(datos['Glucosa'] > 120, 1, 0)

    return pd.DataFrame(datos)

# Generar 1000 datos ficticios con dependencias
num_datos = 1000
datos_dependientes_imc = generar_datos_dependientes_imc(num_datos)

# Guardar en un archivo CSV
datos_dependientes_imc.to_csv('datos_simulados3.csv', index=False)

print("Datos generados y guardados exitosamente.")
