import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import cm
import time

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función de costo para regresión logística
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        # Evitamos log(0)
        f_wb_i = np.maximum(f_wb_i, 1e-15)
        f_wb_i = np.minimum(f_wb_i, 1 - 1e-15)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = cost / m
    return cost

# Función de gradiente con optimización de momento
def compute_gradient(X, y, w, b, v_w=None, v_b=0, beta=0.9):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        error = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    # Aplicar momento si se proporcionan los vectores de velocidad
    if v_w is not None:
        v_w = beta * v_w + (1 - beta) * dj_dw
        v_b = beta * v_b + (1 - beta) * dj_db
        return dj_dw, dj_db, v_w, v_b
    else:
        return dj_dw, dj_db

# Datos proporcionados
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Parámetros mejorados para convergencia rápida
w_init = np.array([1.0, 1.0])  # Mejor punto inicial
b_init = -4.0                  # Mejor punto inicial
alpha = 0.5                    # Tasa de aprendizaje más moderada para evitar oscilaciones
num_iters = 100                # Más de lo necesario, pero por si acaso

# Función principal
def main():
    # Configuración de la figura
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Variables iniciales
    w = w_init.copy()
    b = b_init
    v_w = np.zeros_like(w)  # Velocidad inicial para momento
    v_b = 0                 # Velocidad inicial para momento
    prev_cost = compute_cost(X_train, y_train, w, b)
    converged = False
    precision_threshold = 1e-4  # Umbral un poco más permisivo
    
    # Gráfico 1: Visualización de datos y frontera
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Regresión Logística en Datos Bidimensionales', fontsize=12)
    ax1.set_xlabel('Característica 1', fontsize=10)
    ax1.set_ylabel('Característica 2', fontsize=10)
    ax1.set_xlim(0, 3.5)
    ax1.set_ylim(0, 3)
    ax1.grid(True, alpha=0.3)
    
    # Separamos datos por clase
    X_neg = X_train[y_train == 0]
    X_pos = X_train[y_train == 1]
    
    # Dibujamos puntos
    neg_scatter = ax1.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', marker='o', 
                             s=100, label='Clase 0')
    pos_scatter = ax1.scatter(X_pos[:, 0], X_pos[:, 1], c='red', marker='x', 
                             s=100, linewidths=2, label='Clase 1')
    
    # Dibujamos frontera de decisión inicial
    x1_min, x1_max = ax1.get_xlim()
    x1_grid = np.linspace(x1_min, x1_max, 100)
    decision_boundary, = ax1.plot([], [], 'g-', linewidth=2, label='Frontera de decisión')
    
    # Texto de costo
    cost_value = compute_cost(X_train, y_train, w, b)
    cost_text = ax1.text(0.05, 0.05, f'Costo: {cost_value:.5f}', 
                         transform=ax1.transAxes, fontsize=9, 
                         bbox=dict(facecolor='white', alpha=0.7))
    
    ax1.legend(loc='upper left')
    
    # Función para actualizar la frontera
    def update_decision_boundary():
        if abs(w[1]) > 1e-10:  # Evitar división por cero
            x2_grid = (-w[0] * x1_grid - b) / w[1]
            decision_boundary.set_data(x1_grid, x2_grid)
    
    # Actualizamos frontera inicial
    update_decision_boundary()
    
    # Gráfico 2: Contorno de costo - MEJOR VISUALIZACIÓN
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Contorno de Costo (b constante)', fontsize=12)
    ax2.set_xlabel('w1', fontsize=10)
    ax2.set_ylabel('w2', fontsize=10)
    ax2.set_xlim(-2, 4)
    ax2.set_ylim(-2, 4)
    ax2.grid(True, alpha=0.3)
    
    # Crear malla para contorno
    w1_range = np.linspace(-2, 4, 100)
    w2_range = np.linspace(-2, 4, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = np.zeros((len(w2_range), len(w1_range)))
    
    # Calcular costo para cada punto de la malla
    for i in range(len(w2_range)):
        for j in range(len(w1_range)):
            Z[i, j] = compute_cost(X_train, y_train, np.array([W1[i, j], W2[i, j]]), b)
    
    # Dibujar contornos
    contour = ax2.contour(W1, W2, Z, levels=15, cmap='rainbow')
    current_point, = ax2.plot(w[0], w[1], 'rx', markersize=10)
    
    # Ruta del descenso
    path_w1, path_w2 = [w[0]], [w[1]]
    path_line, = ax2.plot(path_w1, path_w2, 'r-', linewidth=1, alpha=0.7)
    
    # Texto indicativo
    ax2.text(0.7, 0.95, 'Click para elegir w1,w2', transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    # Gráfico 3: Superficie 3D
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    ax3.set_title('Superficie de Costo J(w1, w2)', fontsize=12)
    ax3.set_xlabel('w1', fontsize=10)
    ax3.set_ylabel('w2', fontsize=10)
    ax3.set_zlabel('J(w1, w2)', fontsize=10)
    
    # Superficie de costo
    surf = ax3.plot_surface(W1, W2, Z, cmap=cm.coolwarm, alpha=0.8, antialiased=True)
    current_point_3d, = ax3.plot([w[0]], [w[1]], [compute_cost(X_train, y_train, w, b)], 'rx', markersize=10)
    
    # Trayectoria 3D
    path_J = [compute_cost(X_train, y_train, w, b)]
    path_3d, = ax3.plot([w[0]], [w[1]], [compute_cost(X_train, y_train, w, b)], 'r-', linewidth=1, alpha=0.7)
    
    # Ajustar visión
    ax3.view_init(elev=30, azim=-45)
    
    # Gráfico 4: Progreso del descenso
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('Progreso del Descenso de Gradiente', fontsize=12)
    ax4.set_xlabel('Iteración', fontsize=10)
    ax4.set_ylabel('Costo', fontsize=10)
    ax4.set_xlim(0, 30)  # Solo mostramos las primeras 30 iteraciones
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3)
    
    # Datos iniciales de costo
    iterations = [0]
    costs = [prev_cost]
    cost_line, = ax4.plot(iterations, costs, '-', color='orange', linewidth=2)
    
    # Texto de estado
    status_text = ax4.text(0.05, 0.95, 'Haz clic para iniciar', transform=ax4.transAxes, 
                           fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Botón de ejecución
    button_ax = plt.axes([0.7, 0.41, 0.25, 0.04])
    button = Button(button_ax, 'Ejecutar Descenso de Gradiente', color='orange')
    
    # Botón de detener (nuevo)
    stop_button_ax = plt.axes([0.7, 0.36, 0.25, 0.04])
    stop_button = Button(stop_button_ax, 'Detener', color='lightgray')
    
    # Control de animación
    is_running = [False]
    
    # Actualizar todos los gráficos
    def update_plots():
        # Frontera de decisión
        update_decision_boundary()
        
        # Costo actual
        cost_value = compute_cost(X_train, y_train, w, b)
        cost_text.set_text(f'Costo: {cost_value:.5f}')
        
        # Punto en contorno
        current_point.set_data([w[0]], [w[1]])
        
        # Ruta en contorno
        path_line.set_data(path_w1, path_w2)
        
        # Punto en 3D
        current_point_3d.set_data([w[0]], [w[1]])
        current_point_3d.set_3d_properties([cost_value])
        
        # Ruta en 3D
        path_3d.set_data(path_w1, path_w2)
        path_3d.set_3d_properties(path_J)
        
        # Gráfico de progreso
        # Ajustar automáticamente los límites del eje y si es necesario
        if len(costs) > 0:
            max_cost = max(costs)
            min_cost = min(costs)
            padding = (max_cost - min_cost) * 0.1 if max_cost > min_cost else 0.1
            ax4.set_ylim(max(0, min_cost - padding), max_cost + padding)
        
        # Ajustar automáticamente los límites del eje x
        if len(iterations) > 0:
            ax4.set_xlim(0, max(30, max(iterations) + 5))
            
        cost_line.set_data(iterations, costs)
        
        # Estado actual
        if converged:
            status_text.set_text(f'¡Convergió en {len(iterations)-1} iteraciones! Costo: {cost_value:.5f}')
        else:
            status_text.set_text(f'Iteración: {len(iterations)-1}, Costo: {cost_value:.5f}')
        
        # Actualizar canvas
        fig.canvas.draw_idle()
        plt.pause(0.001)
    
    # Un paso de descenso de gradiente CON ACELERACIÓN
    def gradient_step():
        nonlocal w, b, v_w, v_b, prev_cost, converged
        
        # Calcular gradientes con momento
        dj_dw, dj_db, v_w, v_b = compute_gradient(X_train, y_train, w, b, v_w, v_b, beta=0.9)
        
        # Actualizar parámetros con momento para aceleración
        w = w - alpha * v_w
        b = b - alpha * v_b
        
        # Actualizar trayectoria
        path_w1.append(w[0])
        path_w2.append(w[1])
        
        # Calcular nuevo costo
        current_cost = compute_cost(X_train, y_train, w, b)
        path_J.append(current_cost)
        cost_diff = abs(prev_cost - current_cost)
        
        # Verificar convergencia - más consistente ahora
        if cost_diff < precision_threshold or len(iterations) >= 100:
            converged = True
        
        # Actualizar para próxima iteración
        prev_cost = current_cost
        
        # Actualizar datos de gráficos
        iterations.append(len(iterations))
        costs.append(current_cost)
        
        update_plots()
        
        return converged
    
    # Manejador para detener la ejecución
    def stop_gradient_descent(event):
        is_running[0] = False
        button.label.set_text('Ejecutar Descenso de Gradiente')
    
    stop_button.on_clicked(stop_gradient_descent)
    
    # Manejador del botón
    def run_gradient_descent(event):
        if is_running[0]:
            return
        
        nonlocal converged, w, b, v_w, v_b, prev_cost
        converged = False
        
        # Reiniciar a valores iniciales si ya ha convergido antes
        if len(iterations) > 1:
            w = w_init.copy()
            b = b_init
            v_w = np.zeros_like(w)
            v_b = 0
            prev_cost = compute_cost(X_train, y_train, w, b)
            
            # Resetear gráficos
            iterations.clear()
            costs.clear()
            path_w1.clear()
            path_w2.clear()
            path_J.clear()
            
            iterations.append(0)
            costs.append(prev_cost)
            path_w1.append(w[0])
            path_w2.append(w[1])
            path_J.append(prev_cost)
            
            update_plots()
            
        is_running[0] = True
        button.label.set_text('Ejecutando...')
        
        # Ejecutar hasta convergencia o máximo de iteraciones
        max_iter = 100  # Aumentar el máximo para asegurar convergencia
        current_iter = 0
        while not converged and is_running[0] and current_iter < max_iter:
            converged = gradient_step()
            current_iter += 1
            # Pequeña pausa para visualización
            plt.pause(0.05)  # Pausa más corta para una animación más fluida

        is_running[0] = False
        button.label.set_text('Ejecutar Descenso de Gradiente')
    
    button.on_clicked(run_gradient_descent)
    
    # Clic en el contorno
    def on_click(event):
        nonlocal w, b, prev_cost, converged, path_w1, path_w2, path_J, v_w, v_b
        
        if event.inaxes == ax2 and not is_running[0]:
            # Reiniciar desde punto elegido
            w = np.array([event.xdata, event.ydata])
            v_w = np.zeros_like(w)
            v_b = 0
            path_w1 = [w[0]]
            path_w2 = [w[1]]
            path_J = [compute_cost(X_train, y_train, w, b)]
            
            prev_cost = compute_cost(X_train, y_train, w, b)
            converged = False
            
            # Resetear gráficos
            iterations.clear()
            costs.clear()
            iterations.append(0)
            costs.append(prev_cost)
            
            update_plots()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Ajustar espaciado
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92, wspace=0.2, hspace=0.3)
    
    plt.show()

# Ejecutar programa
if __name__ == "__main__":
    main()