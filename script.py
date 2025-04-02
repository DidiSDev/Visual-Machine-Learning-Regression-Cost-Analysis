import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# estilo de matplotlib default
plt.style.use('default')

# Datos de viviendas (como ejemplo)
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])

# Función para calcular el costo
def compute_cost(x, y, w, b):
    """
    Computa la función de costo para regresión lineal.
    """
    m = x.shape[0]
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    
    total_cost = (1 / (2 * m)) * cost
    return total_cost

# Función para predecir precios
def predict(x, w, b):
    return w * x + b

# parámetros iniciales a ojo
initial_w = 200
initial_b = 0

# calcular el costo inicial
initial_cost = compute_cost(x_train, y_train, initial_w, initial_b)

# Configurar la figura con 3 subplots
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Regresión Lineal - Precios de Viviendas', fontsize=16)

# 1. Gráfico de precios de viviendas
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('Housing Prices')
ax1.set_xlabel('Size (1000 sqft)')
ax1.set_ylabel('Price (in 1000s of dollars)')
ax1.set_xlim(0.8, 3.5)
ax1.set_ylim(200, 750)

# Crear elementos iniciales para el primer gráfico
prediction_line, = ax1.plot([], [], 'b-', linewidth=3, label='Our Prediction')
actual_points = ax1.plot(x_train, y_train, 'rx', markersize=10, label='Actual Value')
cost_lines = []
cost_labels = []

# 2. Gráfico de contorno de costo
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Cost(w,b)')
ax2.set_xlabel('w')
ax2.set_ylabel('b')
ax2.set_xlim(-100, 500)
ax2.set_ylim(-200, 300)

# Crear el gráfico de contorno
w_vals = np.linspace(-100, 500, 100)
b_vals = np.linspace(-200, 300, 100)
W, B = np.meshgrid(w_vals, b_vals)
Z = np.zeros_like(W)

# Calcular valores de costo para el gráfico de contorno (aproximación simplificada)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = compute_cost(x_train, y_train, W[i, j], B[i, j])

# Dibujar contornos
contour = ax2.contour(W, B, Z, levels=[500, 1000, 2000, 4000, 8000, 16000], 
                       colors=['orange', 'blue', 'purple', 'magenta', 'red', 'darkred'])
current_point, = ax2.plot(initial_w, initial_b, 'bo', markersize=10)
cost_label = ax2.text(initial_w + 20, initial_b, f'Cost: {initial_cost:.0f}')

# Añadir recuadro con instrucción
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax2.text(200, 280, 'Click to choose w,b', fontsize=10, bbox=props, ha='center')

# 3. Gráfico 3D de la superficie de costo
ax3 = fig.add_subplot(2, 1, 2, projection='3d')
ax3.set_title('Cost(w,b)\n[You can rotate this figure]')
ax3.set_xlabel('w')
ax3.set_ylabel('b')
ax3.set_zlabel('Cost(w,b)')

# Crear la superficie 3D (usando un rango más limitado para mejor visualización)
w_vals_3d = np.linspace(0, 400, 50)
b_vals_3d = np.linspace(-150, 200, 50)
W_3d, B_3d = np.meshgrid(w_vals_3d, b_vals_3d)
Z_3d = np.zeros_like(W_3d)

for i in range(W_3d.shape[0]):
    for j in range(W_3d.shape[1]):
        Z_3d[i, j] = compute_cost(x_train, y_train, W_3d[i, j], B_3d[i, j])

# Dibujar la superficie 3D con colores que van desde azul (bajo) hasta rojo (alto)
surf = ax3.plot_surface(W_3d, B_3d, Z_3d, cmap=cm.coolwarm, alpha=0.8, linewidth=0)

# Crear el punto 3D inicial - usando un objeto scatter
point3d = ax3.scatter([initial_w], [initial_b], [initial_cost], color='purple', s=50)

# Añadir sliders para w y b
ax_w = plt.axes([0.25, 0.01, 0.50, 0.02])
w_slider = Slider(ax_w, 'w', -100, 500, valinit=initial_w)

ax_b = plt.axes([0.25, 0.04, 0.50, 0.02])
b_slider = Slider(ax_b, 'b', -200, 300, valinit=initial_b)

# Función para actualizar todos los gráficos
def update_all(w, b):
    # Calcular el costo actual
    cost = compute_cost(x_train, y_train, w, b)
    
    # Actualizar el gráfico de precios de viviendas
    x_range = np.array([0.8, 3.5])
    y_range = w * x_range + b
    prediction_line.set_data(x_range, y_range)
    
    # Limpiar líneas de costo previas
    for line in cost_lines:
        line.remove()
    for label in cost_labels:
        label.remove()
    
    cost_lines.clear()
    cost_labels.clear()
    
    # Dibujar líneas punteadas de error y etiquetas
    cost_string = "cost = (1/m)*("
    for i in range(len(x_train)):
        predicted = w * x_train[i] + b
        error_line, = ax1.plot([x_train[i], x_train[i]], [y_train[i], predicted], 
                          'purple', linestyle='--', alpha=0.7)
        error_squared = (predicted - y_train[i])**2
        cost_string += f"{error_squared:.0f} + "
        
        # Etiqueta para el error cuadrado
        error_label = ax1.text(x_train[i] + 0.1, (y_train[i] + predicted)/2, f"{error_squared:.0f}", 
                          color='purple', fontsize=9)
        
        cost_lines.append(error_line)
        cost_labels.append(error_label)
    
    cost_string = cost_string[:-3] + f") = {cost:.0f}"
    cost_formula = ax1.text(1.0, 220, cost_string, color='purple', fontsize=9)
    cost_labels.append(cost_formula)
    
    # Actualizar punto en el gráfico de contorno
    current_point.set_data([w], [b])
    cost_label.set_position((w + 20, b))
    cost_label.set_text(f'Cost: {cost:.0f}')
    
    # Actualizar punto en la superficie 3D - corregido para usar _offsets3d
    point3d._offsets3d = ([w], [b], [cost])
    
    fig.canvas.draw_idle()

# Función para manejar el cambio en los sliders
def update_from_slider(val):
    w = w_slider.val
    b = b_slider.val
    update_all(w, b)

# Conectar el evento de cambio de slider
w_slider.on_changed(update_from_slider)
b_slider.on_changed(update_from_slider)

# Función para manejar clics en el gráfico de contorno
def on_click(event):
    if event.inaxes == ax2:
        w = event.xdata
        b = event.ydata
        w_slider.set_val(w)
        b_slider.set_val(b)
        update_all(w, b)

# Conectar el evento de clic
fig.canvas.mpl_connect('button_press_event', on_click)

# Inicializar visualización
update_all(initial_w, initial_b)

# Leyenda para el primer gráfico
ax1.legend()

# Ajustar espaciado
plt.tight_layout(rect=[0, 0.08, 1, 0.95])

# Mostrar el gráfico y terminamos
plt.show()