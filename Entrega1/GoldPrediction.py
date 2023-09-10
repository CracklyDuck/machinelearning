import numpy as np
import pandas as pd

# Funcion para convertir fechas a numeros enteros


def dateToInt(date):
    number = int(date.strftime("%Y%m"))
    return number/10000


def f1(x, m, b):  # y = m*x + b (ground truth function)
    return m*x + b

# Funcion para calcular el error cuadratico medio


def update_w_and_b(X, y, w, b, alpha):
    '''Update parameters w and b during 1 epoch'''
    dl_dw = 0.0
    dl_db = 0.0
    N = len(X)
    for i in range(N):
        dl_dw += -2*X[i]*(y[i] - (w*X[i] + b))
        dl_db += -2*(y[i] - (w*X[i] + b))
    # update w and b
    w = w - (1/float(N))*dl_dw*alpha
    b = b - (1/float(N))*dl_db*alpha
    return w, b

# Funcion para entrenar el modelo


def train(X, y, w, b, alpha, epochs):
    '''Loops over multiple epochs and prints progress'''
    print('Training progress:')
    for e in range(epochs):
        w, b = update_w_and_b(X, y, w, b, alpha)
    # log the progress
        if e % 400 == 0:
            avg_loss_ = avg_loss(X, y, w, b)
            # print("epoch: {} | loss: {}".format(e, avg_loss_))
            print("Epoch {} | Loss: {} | w:{}, b:{}".format(
                e, avg_loss_, round(w, 4), round(b, 4)))
    print('Training finished.', '\n')
    return w, b


# Funcion para calcular la perdida media
def avg_loss(X, y, w, b):
    '''Calculates the MSE'''
    n = len(X)
    total_error = 0.0
    for i in range(n):
        total_error += (y[i] - (w*X[i] + b))**2
    return total_error / float(n)

# Funcion para predecir


def predict(x, w, b):
    return w*x + b


if __name__ == '__main__':

    # Cargar datos
    df = pd.read_csv('gold prices.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    # Convertir fechas a numeros enteros
    df['Date'] = df['Date'].apply(lambda x: dateToInt(x))
    dfGold = df[['Date', 'Open']]
    dfGold = dfGold.groupby('Date')
    dfGold = dfGold.aggregate({'Open': np.mean})

    # Definir variables
    xmin, xmax, npts = [0, 50, 200]
    X = pd.Series(dfGold.index)
    y0 = pd.Series(dfGold['Open'].tolist())

    w = 0.0
    b = -1000.0
    alpha = 0.001
    epochs = 12000
    # Train Model
    w, b = train(X=X, y=y0, w=0.0, b=0.0, alpha=0.001, epochs=epochs)

    # Usar el modelo entrenado para predecir
    x_new = 20.14
    y_new = predict(x_new, w, b)
    print('Para x={}, la predicción de y es y={}'.format(x_new, round(y_new, 4)))


'''
La actividad me parecio muy interesante, ya que nos permite entender como funciona el algoritmo de regresion lineal, y como se puede 
implementar en python. Escogi este dataset ya que es reciente, y me parecio interesante ver como se comporta el precio del oro en el tiempo. 

Esta actividad tambien me permitio entender como se puede entrenar un modelo; como hacer que porgresivamente mejore y observar como va 
"aprendiendo". Sin olvidar que me pude entender mejor como se puede calcular el error cuadratico medio, y como se puede calcular la 
perdida media.

Finalmente, mi modelo no pudo alcanzar una prediccion respetable. Creo que se debe a que la gradiente descendiente no pudo encontrar un 
minimo local, y por eso no pudo encontrar una recta que se ajustara a los datos. Sin embargo, creo que si se le diera un pequeño ajuste, 
podria encontrar una recta que se ajuste a los datos.
'''
