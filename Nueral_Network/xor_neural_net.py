import numpy as np
import matplotlib.pyplot as plt
# --- 1. ACTIVATION FUNCTION & ITS DERIVATIVE ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig):
    # derivative of sigmoid given the already-computed sigmoid output
    return sig * (1 - sig)


# --- 2. TRAINING FUNCTION ---
def train_xor(learning_rate=0.5, epochs=10000):
    # XOR Dataset
    X = np.array([[0, 0], 
                  [0, 1], 
                  [1, 0], 
                  [1, 1]])
    
    y = np.array([[0], [1], [1], [0]]) # Expected targets
    # y = np.array([[0], [0], [0], [1]]) # Expected targets for AND operation

    np.random.seed(42) # For reproducible random weights

    # Initialize Weights and Biases randomly
    # Input size = 2, Hidden size = 2, Output size = 1
    W1 = np.random.uniform(-1, 1, (2, 2))  # Weights for hidden layer
    b1 = np.zeros((1, 2))                  # Biases for hidden layer
    
    W2 = np.random.uniform(-1, 1, (2, 1))  # Weights for output layer
    b2 = np.zeros((1, 1))                  # Bias for output layer

    print(f"Training with Learning Rate: {learning_rate}, Epochs: {epochs}...\n")

    # --- TRAINING LOOP ---
    for epoch in range(epochs):
        # --- FORWARD PASS ---
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        
        z2 = np.dot(a1, W2) + b2
        pred = sigmoid(z2)
        
        # --- LOSS CALCULATION (MSE) ---
        error = pred - y
        loss = np.mean(np.square(error))
        loss_history = [] if epoch == 0 else loss_history
        loss_history.append(loss)
        
        # --- BACKWARD PASS (Calculus & Chain Rule) ---
        # Output layer gradients
        d_pred = error * sigmoid_derivative(pred)
        
        # Hidden layer gradients
        d_hidden = np.dot(d_pred, W2.T) * sigmoid_derivative(a1)
        
        # --- WEIGHT & BIAS UPDATES (Gradient Descent) ---
        W2 -= learning_rate * np.dot(a1.T, d_pred)
        b2 -= learning_rate * np.sum(d_pred, axis=0, keepdims=True)
        
        W1 -= learning_rate * np.dot(X.T, d_hidden)
        b1 -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

        # Print progress every 2000 epochs
        if (epoch + 1) % 2000 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f}")

    print("\nTraining Finished!")
    
    # Return the trained parameters (weights and biases)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}, loss_history


# --- 3. INFERENCE / PREDICTION FUNCTION ---
def predict(model, X_test):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    
    a1 = sigmoid(np.dot(X_test, W1) + b1)
    pred = sigmoid(np.dot(a1, W2) + b2)
    return pred


if __name__ == "__main__":
    # # Train the model with custom parameters
    # trained_weights = train_xor(learning_rate=0.5, epochs=10000)

    # # Test the trained model
    
    # Train the model and get loss history
    # trained_weights, loss_history = train_xor(learning_rate=0.5, epochs=10000)
    trained_weights = None
    try:
        model_npz = np.load("xor_model.npz")
        trained_weights = {
            "W1": model_npz["W1"],
            "b1": model_npz["b1"],
            "W2": model_npz["W2"],
            "b2": model_npz["b2"]
        }
        print("Model loaded from 'xor_model.npz'!")
    except FileNotFoundError:
        print("Model file not found. Training a new model...")
        trained_weights, loss_history = train_xor(learning_rate=0.5, epochs=10000)
        # Save the model
        np.savez("xor_model.npz", W1=trained_weights["W1"], b1=trained_weights["b1"], 
                               W2=trained_weights["W2"], b2=trained_weights["b2"])
        print("Model saved to 'xor_model.npz'!")

    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = predict(trained_weights, X_test)
    
    print("\nFinal Predictions:")
    for i in range(len(X_test)):
        print(f"Input: {X_test[i]} => Predicted: {predictions[i][0]:.4f} (Rounded: {round(predictions[i][0])})")
        
    # --- 3. PLOTTING THE LEARNING CURVE ---
    # plt.figure(figsize=(8, 5))
    # plt.plot(loss_history, color='blue', linewidth=2)
    # plt.title("Training Loss Over Epochs (XOR Problem)")
    # plt.xlabel("Epoch")
    # plt.ylabel("Mean Squared Error (Loss)")
    # plt.grid(True)
    # plt.show()  # Opens a window displaying the loss curve graph