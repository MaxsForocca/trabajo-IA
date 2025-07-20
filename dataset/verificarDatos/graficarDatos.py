import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = os.path.join(os.path.dirname(__file__), '..\dataset_med.csv')
if not os.path.exists(dataset_path):
    print(f"Dataset path '{dataset_path}' does not exist. Please check the path.")
    sys.exit(1) 
else:
    print(f"Dataset path is set to: {dataset_path}")
    
# age: 
# gender: (masculino, femenino) (igualmente distribuidos)
# estado del cancer hay solo 4 ()
# historial familiar solo hay 2 (si/no) (igualmente distribuidos)
# tabaquismo hay 4 (passive smoker, former smoker, never smoked, current smoker) (igualmente distribuidos)
# nivel de colesterol es variado (muy distribuido) ** 
# Hipertension (muy distribuido, irregular mente bajo) **
# Asma (muy distribuido, irregular?) ** (posible 1/0)
# Cirrosis (muy distribuido, irregular?) ** (posible 1/0)
# Otro cancer (muy distribuido, irregular?) ** (posible 1/0)
# Tipo de tratamiento hay 4 (quimioterapia, cirugía, combinado, radiacion) (igualmente distribuidos)
# Supervivencia (1/0) (muy distribuido, mayormente 0)

df = pd.read_csv(dataset_path, header = 0, usecols=['id', 'age', 'gender', 'cancer_stage', 'family_history', 'smoking_status', 'bmi',
                                                    'cholesterol_level', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer',
                                                    'treatment_type', 'survived'])
    
    
# Asumiendo que ya tienes tu DataFrame df
cols = df.columns
n_cols = 4  # Número de gráficos por fila
n_rows = (len(cols) + n_cols - 1) // n_cols  # Número de filas necesarias

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()  # Para indexar fácilmente incluso si es 1 fila

for i, col in enumerate(cols):
    ax = axes[i]
    unique_vals = df[col].dropna().unique()
    
    if pd.api.types.is_numeric_dtype(df[col]) and set(unique_vals).issubset({0,1}):
        # Columna binaria 0/1 → gráfica conteo como categórica
        sns.countplot(x=col, data=df, ax=ax)
        ax.set_title(f'Conteo binario {col}')
    
    elif pd.api.types.is_numeric_dtype(df[col]):
        # Numérica continua → histograma
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f'Histograma {col}')
    
    else:
        # Categórica → gráfico de barras horizontal
        sns.countplot(y=col, data=df, ax=ax)
        ax.set_title(f'Conteo categorías {col}')
    
    ax.tick_params(axis='x', rotation=45)

# Si quedan ejes vacíos, los ocultamos
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()