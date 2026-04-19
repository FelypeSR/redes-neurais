import math
import random
# Dataset XOR
dataset = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]
#eu tive uma perda de neuronios/ mas optei por 2 mesmo
def relu(x):
   # return max(0, x)
    return x if x > 0 else 0.01 * x


def relu_deriv(x):
    #return 1 if x > 0 else 0
    return 1 if x > 0 else 0.01

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def bce(y, y_hat):
    epsilon = 1e-15
    y_hat = max(min(y_hat, 1 - epsilon), epsilon)
    return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))

def rand():
    return random.uniform(-1, 1)
    #return random.uniform(-0.5, 0.5)
# camada oculta (2 neurônios)
w1 = [rand(), rand()]
w2 = [rand(), rand()]
b1 = rand() # eu poderia começar com 0.2 mas quero deixar random
b2 = rand()
# camada de saída
w_out = [rand(), rand()]
b_out = rand()
# Treinamento
lr = 0.1 #deixei o LR baixo porque em alguns momentos eles estavam tendo resultados ruins
epochs = 10000

for epoch in range(epochs):
    total_loss = 0
    random.shuffle(dataset) #tu não vai ficar decorando não paizão

    for x, y in dataset:
        # FORWARD/camada oculta
        z1 = x[0]*w1[0] + x[1]*w1[1] + b1
        z2 = x[0]*w2[0] + x[1]*w2[1] + b2

        h1 = relu(z1)
        h2 = relu(z2)

        # saída
        z_out = h1*w_out[0] + h2*w_out[1] + b_out
        y_hat = sigmoid(z_out)

        # loss
        loss = bce(y, y_hat)
        total_loss += loss
        # BACKPROP /saída (BCE + sigmoid simplifica!)
        dL_dz_out = y_hat - y
        # gradientes saída
        dw_out0 = dL_dz_out * h1
        dw_out1 = dL_dz_out * h2
        db_out_grad = dL_dz_out
        # gradientes camada oculta
        dL_dh1 = dL_dz_out * w_out[0]
        dL_dh2 = dL_dz_out * w_out[1]
        dL_dz1 = dL_dh1 * relu_deriv(z1)
        dL_dz2 = dL_dh2 * relu_deriv(z2)
        dw1_0 = dL_dz1 * x[0]
        dw1_1 = dL_dz1 * x[1]
        db1_grad = dL_dz1
        dw2_0 = dL_dz2 * x[0]
        dw2_1 = dL_dz2 * x[1]
        db2_grad = dL_dz2

        # saída
        w_out[0] -= lr * dw_out0
        w_out[1] -= lr * dw_out1
        b_out -= lr * db_out_grad
        # oculta
        w1[0] -= lr * dw1_0
        w1[1] -= lr * dw1_1
        b1 -= lr * db1_grad
        w2[0] -= lr * dw2_0
        w2[1] -= lr * dw2_1
        b2 -= lr * db2_grad
    # mostrar evolução
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/4}")
# Teste final
print("\nResultados finais:")
for x, y in dataset:
    z1 = x[0]*w1[0] + x[1]*w1[1] + b1
    z2 = x[0]*w2[0] + x[1]*w2[1] + b2

    h1 = relu(z1)
    h2 = relu(z2)

    z_out = h1*w_out[0] + h2*w_out[1] + b_out
    y_hat = sigmoid(z_out)

    print(f"{x} -> {y_hat:.4f} (esperado: {y})")