# from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import make_regression
# import numpy as np
# import matplotlib.pyplot as plt

# # Reproducibility
# np.random.seed(42)

# # Generate sample regression data with only 2 features for 3D plotting
# X, y = make_regression(n_samples=100, n_features=2, noise=15)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train Lasso
# lasso = Lasso(alpha=1.0)
# lasso.fit(X_train, y_train)

# # Predictions
# y_pred = lasso.predict(X_test)

# # Output
# print("Coefficients:", lasso.coef_)
# print('MSE:', mean_squared_error(y_test, y_pred))

# # Extract test features
# x1_test = X_test[:, 0]
# x2_test = X_test[:, 1]

# # Plot
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x1_test, x2_test, y_test, color='blue', label='Actual', alpha=0.8)
# ax.scatter(x1_test, x2_test, y_pred, color='red', label='Predicted', alpha=0.8)

# # Create prediction surface
# x1_range = np.linspace(x1_test.min(), x1_test.max(), 20)
# x2_range = np.linspace(x2_test.min(), x2_test.max(), 20)
# x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
# X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))
# y_grid = lasso.predict(X_grid)

# # Surface plot
# ax.plot_trisurf(x1_grid.flatten(), x2_grid.flatten(), y_grid,
#                 color='green', alpha=0.5)

# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Target')
# ax.set_title('Lasso Regression Predictions vs Actual Values')
# ax.legend()
# plt.show()


# from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import make_regression
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Reproducibility
# np.random.seed(42)

# # Generate sample regression data with only 2 features for 3D plotting
# X, y = make_regression(n_samples=100, n_features=2, noise=15)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train Lasso
# lasso = Lasso(alpha=1.0)
# lasso.fit(X_train, y_train)

# # Predictions
# y_pred = lasso.predict(X_test)

# # Output
# print("Coefficients:", lasso.coef_)
# print('MSE:', mean_squared_error(y_test, y_pred))

# # Extract test features
# x1_test = X_test[:, 0]
# x2_test = X_test[:, 1]

# # Create prediction surface
# x1_range = np.linspace(x1_test.min(), x1_test.max(), 20)
# x2_range = np.linspace(x2_test.min(), x2_test.max(), 20)
# x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
# X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))
# y_grid = lasso.predict(X_grid)

# # Plot setup
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter points
# ax.scatter(x1_test, x2_test, y_test, color='blue', label='Actual', alpha=0.8)
# ax.scatter(x1_test, x2_test, y_pred, color='red', label='Predicted', alpha=0.8)

# # Surface plot
# surf = ax.plot_trisurf(x1_grid.flatten(), x2_grid.flatten(), y_grid,
#                        color='green', alpha=0.5)

# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Target')
# ax.set_title('Lasso Regression Predictions vs Actual Values')
# ax.legend()

# # Rotation function
# def rotate(angle):
#     ax.view_init(elev=20, azim=angle)

# # Create animation (0 to 360 degrees)
# ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50)

# # Show animation
# plt.show()

# # If you want to save as GIF (uncomment below)
# # ani.save("lasso_rotation.gif", writer="pillow")


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Reproducibility
np.random.seed(42)

# Generate regression data with only 2 features
X, y = make_regression(n_samples=100, n_features=2, noise=15)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# Predictions
y_pred = lasso.predict(X_test)

# Output
print("Coefficients:", lasso.coef_)
print('MSE:', mean_squared_error(y_test, y_pred))

# Test features
x1_test = X_test[:, 0]
x2_test = X_test[:, 1]

# Create prediction surface
x1_range = np.linspace(x1_test.min(), x1_test.max(), 20)
x2_range = np.linspace(x2_test.min(), x2_test.max(), 20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))
y_grid = lasso.predict(X_grid)

# Plot setup
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('Lasso Regression Predictions vs Actual Values')

# Empty plots (will fill in animation)
actual_scatter = ax.scatter([], [], [], color='blue', label='Actual', alpha=0.8)
pred_scatter = ax.scatter([], [], [], color='red', label='Predicted', alpha=0.8)
surf_plot = None
ax.legend()

# Animation update function
def update(frame):
    global surf_plot
    ax.view_init(elev=20, azim=frame)  # rotation

    # Number of points to display gradually
    points_to_show = min(frame, len(x1_test))

    # Update scatter points
    actual_scatter._offsets3d = (
        x1_test[:points_to_show],
        x2_test[:points_to_show],
        y_test[:points_to_show]
    )
    pred_scatter._offsets3d = (
        x1_test[:points_to_show],
        x2_test[:points_to_show],
        y_pred[:points_to_show]
    )

    # Show surface only after points are fully shown
    if points_to_show == len(x1_test) and surf_plot is None:
        surf_plot = ax.plot_trisurf(
            x1_grid.flatten(), x2_grid.flatten(), y_grid,
            color='green', alpha=0.5
        )

# Frames = points appearing + rotation
frames = len(x1_test) + 180  # points + rotation
ani = FuncAnimation(fig, update, frames=frames, interval=50, repeat=True)

plt.show()
