import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuracion de estilo para graficos mas profesionales
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MedicalDataAnalyzer:
    """
    Clase principal para realizar analisis exploratorio de datos medicos
    Incluye metodos para carga, visualizacion y analisis estadistico
    """
    
    def __init__(self, dataset_path=None):
        self.df = None
        self.dataset_path = dataset_path
        # Diccionario para traducir nombres de columnas
        self.column_translations = {
            'id': 'ID',
            'age': 'Edad',
            'gender': 'Genero',
            'cancer_stage': 'Etapa del Cancer',
            'family_history': 'Historial Familiar',
            'smoking_status': 'Estado de Tabaquismo',
            'bmi': 'IMC',
            'cholesterol_level': 'Nivel de Colesterol',
            'hypertension': 'Hipertension',
            'asthma': 'Asma',
            'cirrhosis': 'Cirrosis',
            'other_cancer': 'Otro Cancer',
            'treatment_type': 'Tipo de Tratamiento',
            'survived': 'Supervivencia'
        }
        
    def load_data(self):
        """
        Carga los datos del dataset desde archivo CSV o crea datos sinteticos
        Si el archivo no existe, genera un dataset de ejemplo para demostracion
        Incluye validacion de ruta y manejo de errores
        """
        try:
            if self.dataset_path and os.path.exists(self.dataset_path):
                self.df = pd.read_csv(self.dataset_path, header=0, 
                                    usecols=['id', 'age', 'gender', 'cancer_stage', 
                                           'family_history', 'smoking_status', 'bmi',
                                           'cholesterol_level', 'hypertension', 'asthma', 
                                           'cirrhosis', 'other_cancer', 'treatment_type', 'survived'])
                print(f"Dataset cargado exitosamente desde: {self.dataset_path}")
            else:
                # Opcion alternativa: usar dataset de ejemplo
                print("Archivo no encontrado. Usando dataset de ejemplo...")
                # Crear dataset sintetico para demostracion
                np.random.seed(42)
                n_samples = 1000
                self.df = pd.DataFrame({
                    'id': range(1, n_samples + 1),
                    'age': np.random.normal(60, 15, n_samples).astype(int),
                    'gender': np.random.choice(['Male', 'Female'], n_samples),
                    'cancer_stage': np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'], n_samples),
                    'family_history': np.random.choice(['Yes', 'No'], n_samples),
                    'smoking_status': np.random.choice(['Never', 'Former', 'Current', 'Passive'], n_samples),
                    'bmi': np.random.normal(25, 5, n_samples),
                    'cholesterol_level': np.random.normal(200, 40, n_samples),
                    'hypertension': np.random.choice([0, 1], n_samples),
                    'asthma': np.random.choice([0, 1], n_samples),
                    'cirrhosis': np.random.choice([0, 1], n_samples),
                    'other_cancer': np.random.choice([0, 1], n_samples),
                    'treatment_type': np.random.choice(['Chemotherapy', 'Surgery', 'Radiation', 'Combined'], n_samples),
                    'survived': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
                })
                
        except Exception as e:
            print(f"Error cargando datos: {e}")
            sys.exit(1)
    
    def basic_info(self):
        """
        Muestra informacion basica del dataset incluyendo:
        - Dimensiones (filas y columnas)
        - Uso de memoria
        - Tipos de datos de cada columna
        - Valores faltantes por columna con porcentajes
        - Estadisticas descriptivas para variables numericas y categoricas
        """
        print("="*60)
        print("INFORMACION BASICA DEL DATASET")
        print("="*60)
        print(f"Dimensiones: {self.df.shape[0]} filas x {self.df.shape[1]} columnas")
        print(f"Memoria utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("\nTIPOS DE DATOS:")
        print(self.df.dtypes)
        
        print("\nVALORES FALTANTES:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Columna': missing_data.index,
            'Valores Faltantes': missing_data.values,
            'Porcentaje (%)': missing_percent.values
        })
        print(missing_df[missing_df['Valores Faltantes'] > 0])
        
        print("\nESTADISTICAS DESCRIPTIVAS:")
        print(self.df.describe(include='all').round(2))
    
    def improved_visualization(self):
        """
        Genera visualizaciones automaticas para todas las variables del dataset
        Clasifica las variables en tres tipos:
        - Numericas continuas: genera histogramas con estadisticas
        - Categoricas: genera graficos de pie con distribuciones
        - Binarias (0/1): genera graficos de barras con conteos
        Traduce los nombres de las variables al español para mejor comprension
        """
        # Separar columnas por tipo
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        binary_cols = []
        
        # Identificar columnas binarias
        for col in numeric_cols:
            unique_vals = self.df[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                binary_cols.append(col)
        
        # Remover binarias de numericas
        numeric_cols = [col for col in numeric_cols if col not in binary_cols and col != 'id']
        
        print(f"Variables numericas: {numeric_cols}")
        print(f"Variables categoricas: {categorical_cols}")
        print(f"Variables binarias: {binary_cols}")
        
        # Graficos para variables numericas
        if numeric_cols:
            self._plot_numeric_variables(numeric_cols)
        
        # Graficos para variables categoricas
        if categorical_cols:
            self._plot_categorical_variables(categorical_cols)
        
        # Graficos para variables binarias
        if binary_cols:
            self._plot_binary_variables(binary_cols)
    
    def _plot_numeric_variables(self, numeric_cols):
        """
        Genera histogramas para variables numericas continuas
        Incluye estadisticas descriptivas (media y desviacion estandar) en cada grafico
        Utiliza nombres en español para mejor comprension
        Proporciona analisis automatico de cada distribucion
        """
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle('DISTRIBUCION DE VARIABLES NUMERICAS', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        print("\n" + "="*80)
        print("ANALISIS DE VARIABLES NUMERICAS")
        print("="*80)
        
        for i, col in enumerate(numeric_cols):
            ax = axes[i] if len(numeric_cols) > 1 else axes
            
            # Calcular estadisticas
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            median_val = self.df[col].median()
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            skewness = self.df[col].skew()
            
            # Crear histograma con traduccion al español
            spanish_name = self.column_translations.get(col, col)
            self.df[col].hist(bins=30, alpha=0.7, ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'{spanish_name}\nMedia: {mean_val:.2f} | Std: {std_val:.2f}')
            ax.set_xlabel(spanish_name)
            ax.set_ylabel('Frecuencia')
            ax.grid(True, alpha=0.3)
            
            # Agregar linea de media
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Media: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Mediana: {median_val:.2f}')
            ax.legend()
            
            # Analisis automatico
            print(f"\n--- ANALISIS DE {spanish_name.upper()} ---")
            print(f"Estadisticas basicas:")
            print(f"  • Media: {mean_val:.2f}")
            print(f"  • Mediana: {median_val:.2f}")
            print(f"  • Desviacion estandar: {std_val:.2f}")
            print(f"  • Rango: {min_val:.2f} - {max_val:.2f}")
            print(f"  • Asimetria (skewness): {skewness:.2f}")
            
            # Interpretacion de la distribucion
            if abs(skewness) < 0.5:
                distribution_type = "aproximadamente normal (simetrica)"
            elif skewness > 0.5:
                distribution_type = "sesgada hacia la derecha (cola derecha larga)"
            else:
                distribution_type = "sesgada hacia la izquierda (cola izquierda larga)"
            
            print(f"Interpretacion:")
            print(f"  • La distribucion es {distribution_type}")
            
            # Detectar posibles outliers usando IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            
            print(f"  • Posibles valores atipicos: {len(outliers)} ({len(outliers)/len(self.df)*100:.1f}%)")
            
            if abs(mean_val - median_val) < std_val * 0.1:
                print(f"  • La media y mediana son muy similares, indicando distribucion equilibrada")
            elif mean_val > median_val:
                print(f"  • La media es mayor que la mediana, indicando valores altos extremos")
            else:
                print(f"  • La mediana es mayor que la media, indicando valores bajos extremos")
        
        # Ocultar ejes vacios
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    def _plot_categorical_variables(self, categorical_cols):
        """
        Genera graficos de pie para variables categoricas
        Muestra la distribucion porcentual de cada categoria
        Incluye el total de categorias unicas por variable
        Utiliza nombres en español para mejor comprension
        Proporciona analisis de balance y dominancia de categorias
        """
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        fig.suptitle('DISTRIBUCION DE VARIABLES CATEGORICAS', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        print("\n" + "="*80)
        print("ANALISIS DE VARIABLES CATEGORICAS")
        print("="*80)
        
        for i, col in enumerate(categorical_cols):
            ax = axes[i] if len(categorical_cols) > 1 else axes
            
            spanish_name = self.column_translations.get(col, col)
            value_counts = self.df[col].value_counts()
            percentages = (value_counts / len(self.df)) * 100
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
            
            wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index, 
                                            autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title(f'{spanish_name}\n(Total categorias: {len(value_counts)})')
            
            # Analisis automatico
            print(f"\n--- ANALISIS DE {spanish_name.upper()} ---")
            print(f"Distribucion de categorias:")
            for category, count, pct in zip(value_counts.index, value_counts.values, percentages.values):
                print(f"  • {category}: {count} casos ({pct:.1f}%)")
            
            # Analisis de balance
            max_pct = percentages.max()
            min_pct = percentages.min()
            dominant_category = percentages.idxmax()
            
            print(f"Analisis de balance:")
            if max_pct > 70:
                balance_status = "muy desbalanceada"
                print(f"  • La variable esta {balance_status}")
                print(f"  • '{dominant_category}' domina con {max_pct:.1f}% de los casos")
                print(f"  • Esto podria causar sesgo en modelos de Machine Learning")
            elif max_pct > 50:
                balance_status = "moderadamente desbalanceada"
                print(f"  • La variable esta {balance_status}")
                print(f"  • '{dominant_category}' es la categoria mas frecuente ({max_pct:.1f}%)")
            else:
                balance_status = "relativamente balanceada"
                print(f"  • La variable esta {balance_status}")
                print(f"  • Buena distribucion para analisis estadistico")
            
            # Diversidad
            entropy = -sum((p/100) * np.log2(p/100) for p in percentages if p > 0)
            max_entropy = np.log2(len(value_counts))
            diversity_ratio = entropy / max_entropy
            
            print(f"  • Indice de diversidad: {diversity_ratio:.2f} (0=sin diversidad, 1=maxima diversidad)")
            
            if diversity_ratio > 0.8:
                print(f"  • Alta diversidad: las categorias estan bien distribuidas")
            elif diversity_ratio > 0.5:
                print(f"  • Diversidad moderada: algunas categorias dominan")
            else:
                print(f"  • Baja diversidad: una o pocas categorias dominan fuertemente")
        
        # Ocultar ejes vacios
        for j in range(len(categorical_cols), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    def _plot_binary_variables(self, binary_cols):
        """
        Genera graficos de barras para variables binarias (0/1)
        Muestra la frecuencia de cada valor con anotaciones numericas
        Utiliza colores diferentes para cada valor (0 y 1)
        Utiliza nombres en español para mejor comprension
        Proporciona analisis de balance y interpretacion medica
        """
        n_cols = min(4, len(binary_cols))
        n_rows = (len(binary_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        fig.suptitle('DISTRIBUCION DE VARIABLES BINARIAS', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        print("\n" + "="*80)
        print("ANALISIS DE VARIABLES BINARIAS")
        print("="*80)
        
        for i, col in enumerate(binary_cols):
            ax = axes[i] if len(binary_cols) > 1 else axes
            
            spanish_name = self.column_translations.get(col, col)
            counts = self.df[col].value_counts().sort_index()
            colors = ['lightcoral', 'lightblue']
            
            bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)])
            ax.set_title(f'{spanish_name}')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Frecuencia')
            
            # Agregar valores en las barras
            for bar, count in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts.values),
                       f'{count}', ha='center', va='bottom')
            
            # Analisis automatico
            print(f"\n--- ANALISIS DE {spanish_name.upper()} ---")
            
            if len(counts) == 2:
                count_0 = counts.get(0, 0)
                count_1 = counts.get(1, 0)
                total = count_0 + count_1
                pct_0 = (count_0 / total) * 100
                pct_1 = (count_1 / total) * 100
                
                print(f"Distribucion:")
                print(f"  • Valor 0 (No/Ausente): {count_0} casos ({pct_0:.1f}%)")
                print(f"  • Valor 1 (Si/Presente): {count_1} casos ({pct_1:.1f}%)")
                
                # Interpretacion del balance
                ratio = max(pct_0, pct_1) / min(pct_0, pct_1) if min(pct_0, pct_1) > 0 else float('inf')
                
                if ratio < 1.5:
                    balance_interpretation = "muy balanceada"
                    ml_impact = "Excelente para modelos de ML"
                elif ratio < 3:
                    balance_interpretation = "moderadamente balanceada"
                    ml_impact = "Buena para modelos de ML"
                elif ratio < 10:
                    balance_interpretation = "desbalanceada"
                    ml_impact = "Considerar tecnicas de balanceo (SMOTE, undersampling)"
                else:
                    balance_interpretation = "muy desbalanceada"
                    ml_impact = "Requiere tecnicas especiales de balanceo"
                
                print(f"Balance de clases:")
                print(f"  • La variable esta {balance_interpretation}")
                print(f"  • Ratio de desbalance: {ratio:.1f}:1")
                print(f"  • Impacto en ML: {ml_impact}")
                
                # Interpretacion medica especifica
                medical_interpretations = {
                    'hypertension': f"  • {pct_1:.1f}% de pacientes tienen hipertension",
                    'asthma': f"  • {pct_1:.1f}% de pacientes tienen asma",
                    'cirrhosis': f"  • {pct_1:.1f}% de pacientes tienen cirrosis",
                    'other_cancer': f"  • {pct_1:.1f}% de pacientes tienen historial de otro cancer",
                    'survived': f"  • Tasa de supervivencia: {pct_1:.1f}%"
                }
                
                if col in medical_interpretations:
                    print(f"Interpretacion medica:")
                    print(medical_interpretations[col])
                    
                    if col == 'survived':
                        if pct_1 > 70:
                            print(f"  • Tasa de supervivencia alta - pronostico favorable")
                        elif pct_1 > 50:
                            print(f"  • Tasa de supervivencia moderada")
                        else:
                            print(f"  • Tasa de supervivencia baja - requiere atencion")
                    
                    elif pct_1 > 30:
                        print(f"  • Prevalencia alta de esta condicion en el dataset")
                    elif pct_1 > 10:
                        print(f"  • Prevalencia moderada de esta condicion")
                    else:
                        print(f"  • Prevalencia baja de esta condicion")
            else:
                print(f"Distribucion:")
                for value, count in counts.items():
                    pct = (count / len(self.df)) * 100
                    print(f"  • Valor {value}: {count} casos ({pct:.1f}%)")
        
        # Ocultar ejes vacios
        for j in range(len(binary_cols), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self):
        """
        Realiza analisis de correlacion entre variables numericas
        Genera una matriz de correlacion visual usando heatmap
        Identifica y muestra las correlaciones mas fuertes (|r| > 0.5)
        Utiliza mascara triangular para evitar redundancia en la visualizacion
        Incluye traduccion de nombres de variables al español
        Proporciona interpretacion detallada de las correlaciones encontradas
        """
        # Solo variables numericas
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = numeric_df.corr()
            
            # Crear heatmap con anotaciones y nombres traducidos
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            # Traducir nombres de columnas para el heatmap
            translated_columns = [self.column_translations.get(col, col) for col in correlation_matrix.columns]
            correlation_matrix_translated = correlation_matrix.copy()
            correlation_matrix_translated.columns = translated_columns
            correlation_matrix_translated.index = translated_columns
            
            sns.heatmap(correlation_matrix_translated, mask=mask, annot=True, cmap='RdYlBu_r', 
                       center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('MATRIZ DE CORRELACION', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\n" + "="*80)
            print("ANALISIS DE CORRELACIONES")
            print("="*80)
            
            # Mostrar correlaciones mas fuertes
            print("\nCORRELACIONES MAS FUERTES (|r| > 0.5):")
            high_corr = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        var1_spanish = self.column_translations.get(correlation_matrix.columns[i], correlation_matrix.columns[i])
                        var2_spanish = self.column_translations.get(correlation_matrix.columns[j], correlation_matrix.columns[j])
                        high_corr.append((var1_spanish, var2_spanish, corr_val, correlation_matrix.columns[i], correlation_matrix.columns[j]))
            
            if high_corr:
                for var1, var2, corr, orig_var1, orig_var2 in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
                    print(f"\n--- {var1.upper()} <-> {var2.upper()} ---")
                    print(f"Coeficiente de correlacion: {corr:.3f}")
                    
                    # Interpretacion de la fuerza
                    if abs(corr) >= 0.9:
                        strength = "muy fuerte"
                    elif abs(corr) >= 0.7:
                        strength = "fuerte"
                    elif abs(corr) >= 0.5:
                        strength = "moderada"
                    else:
                        strength = "debil"
                    
                    direction = "positiva" if corr > 0 else "negativa"
                    
                    print(f"Interpretacion: Correlacion {direction} {strength}")
                    
                    if corr > 0:
                        print(f"  • Cuando {var1} aumenta, {var2} tiende a aumentar")
                        print(f"  • Relacion directamente proporcional")
                    else:
                        print(f"  • Cuando {var1} aumenta, {var2} tiende a disminuir")
                        print(f"  • Relacion inversamente proporcional")
                    
                    # Interpretaciones medicas especificas
                    medical_correlations = {
                        ('age', 'cholesterol_level'): "Relacion esperada: el colesterol tiende a aumentar con la edad",
                        ('bmi', 'hypertension'): "Relacion conocida: mayor IMC se asocia con mayor riesgo de hipertension",
                        ('smoking_status', 'survived'): "Impacto del tabaquismo en la supervivencia",
                        ('cancer_stage', 'survived'): "Relacion critica: etapa del cancer vs supervivencia",
                        ('age', 'survived'): "Efecto de la edad en la supervivencia del paciente"
                    }
                    
                    correlation_key = tuple(sorted([orig_var1, orig_var2]))
                    if correlation_key in medical_correlations:
                        print(f"Contexto medico: {medical_correlations[correlation_key]}")
                    
                    # Advertencias sobre multicolinealidad
                    if abs(corr) > 0.8:
                        print(f"ADVERTENCIA: Multicolinealidad detectada")
                        print(f"  • Considerar eliminar una de estas variables para ML")
                        print(f"  • Pueden causar problemas en regresion lineal")
                
                # Resumen general
                print(f"\n--- RESUMEN DE CORRELACIONES ---")
                total_pairs = len(correlation_matrix.columns) * (len(correlation_matrix.columns) - 1) // 2
                strong_corr = sum(1 for _, _, corr, _, _ in high_corr if abs(corr) > 0.7)
                moderate_corr = len(high_corr) - strong_corr
                
                print(f"Total de pares de variables analizados: {total_pairs}")
                print(f"Correlaciones fuertes (|r| > 0.7): {strong_corr}")
                print(f"Correlaciones moderadas (0.5 < |r| <= 0.7): {moderate_corr}")
                print(f"Porcentaje de correlaciones significativas: {len(high_corr)/total_pairs*100:.1f}%")
                
                if len(high_corr) / total_pairs > 0.3:
                    print("INTERPRETACION: Alto grado de intercorrelacion entre variables")
                    print("RECOMENDACION: Considerar tecnicas de reduccion de dimensionalidad (PCA)")
                elif len(high_corr) / total_pairs > 0.1:
                    print("INTERPRETACION: Grado moderado de intercorrelacion")
                    print("RECOMENDACION: Monitorear multicolinealidad en modelos lineales")
                else:
                    print("INTERPRETACION: Bajo grado de intercorrelacion")
                    print("RECOMENDACION: Variables relativamente independientes")
                    
            else:
                print("No se encontraron correlaciones fuertes (|r| > 0.5)")
                print("INTERPRETACION: Las variables numericas son relativamente independientes")
                print("VENTAJA: Menor riesgo de multicolinealidad en modelos de ML")

    def generate_summary_report(self):
        """
        Genera un reporte resumen completo del analisis exploratorio
        Incluye hallazgos principales, recomendaciones para limpieza de datos
        y sugerencias para el siguiente paso del pipeline de ML
        """
        print("\n" + "="*80)
        print("REPORTE RESUMEN DEL ANALISIS EXPLORATORIO")
        print("="*80)
        
        # Informacion general
        print("\n1. INFORMACION GENERAL DEL DATASET:")
        print(f"   • Total de registros: {len(self.df)}")
        print(f"   • Total de variables: {len(self.df.columns)}")
        
        # Clasificacion de variables
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        binary_cols = []
        
        for col in numeric_cols:
            unique_vals = self.df[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                binary_cols.append(col)
        
        continuous_cols = [col for col in numeric_cols if col not in binary_cols and col != 'id']
        
        print(f"   • Variables continuas: {len(continuous_cols)}")
        print(f"   • Variables categoricas: {len(categorical_cols)}")
        print(f"   • Variables binarias: {len(binary_cols)}")
        
        # Calidad de datos
        total_missing = self.df.isnull().sum().sum()
        missing_percentage = (total_missing / (len(self.df) * len(self.df.columns))) * 100
        
        print(f"\n2. CALIDAD DE DATOS:")
        print(f"   • Total de valores faltantes: {total_missing} ({missing_percentage:.2f}%)")
        
        if missing_percentage < 1:
            data_quality = "Excelente"
        elif missing_percentage < 5:
            data_quality = "Buena"
        elif missing_percentage < 15:
            data_quality = "Regular"
        else:
            data_quality = "Pobre"
        
        print(f"   • Calidad general de datos: {data_quality}")
        
        # Analisis de outliers para variables continuas
        print(f"\n3. DETECCION DE VALORES ATIPICOS:")
        total_outliers = 0
        for col in continuous_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = len(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)])
            total_outliers += outliers
            
            spanish_name = self.column_translations.get(col, col)
            outlier_pct = (outliers / len(self.df)) * 100
            print(f"   • {spanish_name}: {outliers} outliers ({outlier_pct:.1f}%)")
        
        # Balance de clases para variable objetivo
        if 'survived' in self.df.columns:
            survival_counts = self.df['survived'].value_counts()
            survival_balance = min(survival_counts) / max(survival_counts)
            
            print(f"\n4. ANALISIS DE VARIABLE OBJETIVO (Supervivencia):")
            print(f"   • Casos de supervivencia: {survival_counts.get(1, 0)}")
            print(f"   • Casos de no supervivencia: {survival_counts.get(0, 0)}")
            print(f"   • Balance de clases: {survival_balance:.2f}")
            
            if survival_balance > 0.8:
                balance_status = "Muy balanceado"
                ml_recommendation = "Ideal para modelos de clasificacion"
            elif survival_balance > 0.5:
                balance_status = "Moderadamente balanceado"
                ml_recommendation = "Adecuado para la mayoria de modelos"
            else:
                balance_status = "Desbalanceado"
                ml_recommendation = "Requiere tecnicas de balanceo (SMOTE, cost-sensitive learning)"
            
            print(f"   • Estado: {balance_status}")
            print(f"   • Recomendacion: {ml_recommendation}")
        
        # Recomendaciones para siguientes pasos
        print(f"\n5. RECOMENDACIONES PARA SIGUIENTES PASOS:")
        
        print(f"\n   LIMPIEZA DE DATOS:")
        if total_missing > 0:
            print(f"   • Tratar {total_missing} valores faltantes")
            print(f"   • Para variables numericas: considerar imputacion por mediana")
            print(f"   • Para variables categoricas: considerar imputacion por moda")
        else:
            print(f"   • No se requiere tratamiento de valores faltantes")
        
        if total_outliers > len(self.df) * 0.05:
            print(f"   • Revisar {total_outliers} outliers detectados")
            print(f"   • Considerar transformaciones (log, sqrt) o eliminacion")
        else:
            print(f"   • Nivel aceptable de outliers")
        
        print(f"\n   TRANSFORMACION DE DATOS:")
        print(f"   • Codificar {len(categorical_cols)} variables categoricas (One-Hot, Label Encoding)")
        print(f"   • Normalizar variables continuas (StandardScaler, MinMaxScaler)")
        if len(binary_cols) > 0 and 'survived' in binary_cols:
            print(f"   • Variable objetivo ya esta codificada como binaria")
        
        print(f"\n   MODELAMIENTO:")
        print(f"   • Dataset apto para algoritmos de clasificacion")
        print(f"   • Considerar: Random Forest, XGBoost, SVM, Logistic Regression")
        print(f"   • Usar validacion cruzada con {min(10, len(self.df)//100)} folds")
        
        correlation_matrix = self.df.select_dtypes(include=[np.number]).corr()
        high_corr_count = 0
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    high_corr_count += 1
        
        if high_corr_count > 0:
            print(f"   • ATENCION: {high_corr_count} pares de variables altamente correlacionadas")
            print(f"   • Considerar eliminacion de variables redundantes")
        
        print(f"\n6. METRICAS DE EVALUACION RECOMENDADAS:")
        print(f"   • Accuracy, Precision, Recall, F1-Score")
        print(f"   • ROC-AUC para evaluacion de clasificacion binaria")
        print(f"   • Matriz de confusion para analisis detallado")
        
        print(f"\n" + "="*80)
        print("ANALISIS EXPLORATORIO COMPLETADO EXITOSAMENTE")
        print("SIGUIENTE PASO: Proceder con limpieza y transformacion de datos")
        print("="*80)


def main():
    """
    Funcion principal que ejecuta todo el pipeline de analisis exploratorio
    Configura la ruta del dataset, crea la instancia del analizador
    y ejecuta todos los metodos de analisis en secuencia
    Retorna el DataFrame procesado para uso posterior
    """
    # Configurar ruta del dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '../dataset_med.csv') if '__file__' in globals() else None
    
    # Crear instancia del analizador
    analyzer = MedicalDataAnalyzer(dataset_path)
    
    # Ejecutar analisis
    print("INICIANDO ANALISIS EXPLORATORIO DE DATOS MEDICOS")
    print("="*60)
    
    analyzer.load_data()
    analyzer.basic_info()
    analyzer.improved_visualization()
    analyzer.correlation_analysis()
    analyzer.generate_summary_report()
    
    print("\nANALISIS COMPLETADO")
    return analyzer.df


# Ejecutar analisis
if __name__ == "__main__":
    df = main()