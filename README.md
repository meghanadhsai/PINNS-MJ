
# 📘 Solving the 1D Heat Conduction Equation Using Physics-Informed Neural Networks (PINNs)
This repository demonstrates the application of **Physics-Informed Neural Networks (PINNs)** to solve the **one-dimensional heat conduction equation**. By incorporating the physics of the problem directly into the loss function, the PINN ensures that the neural network respects the governing differential equation while minimizing the data and physics-based loss.

## 🔍 **Problem Overview**

We solve the 1D heat conduction equation, given by:
\[
\frac{d^2 u}{dx^2} = 0
\]
where \( u(x) \) is the temperature distribution along a rod of length \( L = 1.0 \), with the following boundary conditions:
- \( u(0) = 100 \) (at \( x = 0 \))
- \( u(1) = 0 \) (at \( x = L \))

The goal is to predict the temperature distribution along the rod using a neural network that incorporates these boundary conditions and the heat conduction equation.

```

```

- **Input**: Position along the rod \( x \)
- **Output**: Predicted temperature \( u(x) \)
- **Activation**: Exponential Linear Unit (ELU) activation function is used for each hidden layer.

## ⚙️ **Training Process**

The model is trained using the **AdamW** optimizer with a learning rate decay scheduler (`StepLR`). The loss function combines two terms:
1. **Physics Loss**: Enforces the heat conduction equation \( \frac{d^2 u}{dx^2} = 0 \).
2. **Boundary Loss**: Enforces the boundary conditions \( u(0) = 100 \) and \( u(1) = 0 \).

The total loss is adaptively normalized based on the ratio between physics loss and boundary loss.

## 🖼️ **Results**

After training the model, the temperature distribution along the rod is predicted, and the result is visualized as shown below:

## 📈 **Training Performance**

- **Loss**: Displays both the total loss and the individual contributions of the physics-based and boundary condition losses.
- **Normalization Factor**: Adaptive normalization factor ensures stability during training.

## ⚙️ **Dependencies**

- Python 3.x
- PyTorch 1.9+
- NumPy
- Matplotlib



