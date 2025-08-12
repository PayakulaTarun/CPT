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


# from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import make_regression
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Reproducibility
# np.random.seed(42)

# # Generate regression data with only 2 features
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

# # Test features
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

# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Target')
# ax.set_title('Lasso Regression Predictions vs Actual Values')

# # Empty plots (will fill in animation)
# actual_scatter = ax.scatter([], [], [], color='blue', label='Actual', alpha=0.8)
# pred_scatter = ax.scatter([], [], [], color='red', label='Predicted', alpha=0.8)
# surf_plot = None
# ax.legend()

# # Animation update function
# def update(frame):
#     global surf_plot
#     ax.view_init(elev=20, azim=frame)  # rotation

#     # Number of points to display gradually
#     points_to_show = min(frame, len(x1_test))

#     # Update scatter points
#     actual_scatter._offsets3d = (
#         x1_test[:points_to_show],
#         x2_test[:points_to_show],
#         y_test[:points_to_show]
#     )
#     pred_scatter._offsets3d = (
#         x1_test[:points_to_show],
#         x2_test[:points_to_show],
#         y_pred[:points_to_show]
#     )

#     # Show surface only after points are fully shown
#     if points_to_show == len(x1_test) and surf_plot is None:
#         surf_plot = ax.plot_trisurf(
#             x1_grid.flatten(), x2_grid.flatten(), y_grid,
#             color='green', alpha=0.5
#         )

# # Frames = points appearing + rotation
# frames = len(x1_test) + 180  # points + rotation
# ani = FuncAnimation(fig, update, frames=frames, interval=50, repeat=True)

# plt.show()



# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import make_regression
# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(42)

# # Generate regression data (ElasticNet is regression)
# X, y = make_regression(n_samples=100, n_features=2, noise=10)

# # Centering features (optional)
# X = X - np.mean(X, axis=0)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train ElasticNet
# elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
# elastic_net.fit(X_train, y_train)

# # Predictions
# y_pred = elastic_net.predict(X_test)

# # Output
# print("Coefficients:", elastic_net.coef_)
# print("MSE:", mean_squared_error(y_test, y_pred))

# # 3D Plot setup
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter actual points
# ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Actual', alpha=0.8)

# # Scatter predicted points
# ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, color='red', label='Predicted', alpha=0.8)

# # Create surface grid
# x1_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 20)
# x2_range = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 20)
# x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
# X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))
# y_grid = elastic_net.predict(X_grid)

# # Plot regression surface
# ax.plot_trisurf(x1_grid.flatten(), x2_grid.flatten(), y_grid, color='green', alpha=0.5)

# # Labels
# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Target')
# ax.set_title('ElasticNet Regression Predictions vs Actual Values')
# ax.legend()

# plt.show()

# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import make_regression
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# np.random.seed(42)

# # Generate regression data
# X, y = make_regression(n_samples=100, n_features=2, noise=10)

# # Centering features (optional)
# X = X - np.mean(X, axis=0)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train ElasticNet
# elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
# elastic_net.fit(X_train, y_train)

# # Predictions
# y_pred = elastic_net.predict(X_test)

# # Output
# print("Coefficients:", elastic_net.coef_)
# print("MSE:", mean_squared_error(y_test, y_pred))

# # Create figure
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111, projection='3d')

# # Create surface grid
# x1_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 20)
# x2_range = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 20)
# x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
# X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))
# y_grid = elastic_net.predict(X_grid)

# # Plot static data (will remain constant in animation)
# actual_scatter = ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Actual', alpha=0.8)
# pred_scatter = ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, color='red', label='Predicted', alpha=0.8)
# surface = ax.plot_trisurf(x1_grid.flatten(), x2_grid.flatten(), y_grid, color='green', alpha=0.5)

# # Labels
# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Target')
# ax.set_title('ElasticNet Regression Predictions vs Actual Values')
# ax.legend()

# # Animation function
# def rotate(angle):
#     ax.view_init(elev=20, azim=angle)

# # Create animation
# ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=100)

# plt.show()

# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import make_regression
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # === Data Preparation ===
# np.random.seed(42)
# X, y = make_regression(n_samples=100, n_features=2, noise=10)
# X = X - np.mean(X, axis=0)  # Centering
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # === Train ElasticNet ===
# elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
# elastic_net.fit(X_train, y_train)
# y_pred = elastic_net.predict(X_test)

# print("Coefficients:", elastic_net.coef_)
# print("MSE:", mean_squared_error(y_test, y_pred))

# # === Prepare grid for surface ===
# x1_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 20)
# x2_range = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 20)
# x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
# X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))
# y_grid = elastic_net.predict(X_grid)

# # === Create Figure ===
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Target')
# ax.set_title('ElasticNet Regression Predictions vs Actual Values')

# # Empty handles for animation
# actual_scatter = None
# pred_scatter = None
# surface = None

# # === Animation Function ===
# def animate(frame):
#     global actual_scatter, pred_scatter, surface
#     ax.clear()

#     # Keep labels and title after clearing
#     ax.set_xlabel('Feature 1')
#     ax.set_ylabel('Feature 2')
#     ax.set_zlabel('Target')
#     ax.set_title('ElasticNet Regression Predictions vs Actual Values')

#     # Step 1: Actual points
#     if frame >= 0:
#         actual_scatter = ax.scatter(X_test[:, 0], X_test[:, 1], y_test,
#                                     color='blue', label='Actual', alpha=0.8)

#     # Step 2: Predicted points
#     if frame >= 20:
#         pred_scatter = ax.scatter(X_test[:, 0], X_test[:, 1], y_pred,
#                                   color='red', label='Predicted', alpha=0.8)

#     # Step 3: Regression surface
#     if frame >= 40:
#         surface = ax.plot_trisurf(x1_grid.flatten(), x2_grid.flatten(), y_grid,
#                                   color='green', alpha=0.5)

#     # Legend
#     ax.legend(loc='upper left')

#     # Rotation
#     ax.view_init(elev=20, azim=frame * 2)

# # === Run Animation ===
# ani = FuncAnimation(fig, animate, frames=100, interval=100, repeat=True)
# plt.show()

# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import make_regression
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D

# # === Data Preparation ===
# np.random.seed(42)
# X, y = make_regression(n_samples=100, n_features=2, noise=10)
# X = X - np.mean(X, axis=0)  # Centering
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # === Train ElasticNet ===
# elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
# elastic_net.fit(X_train, y_train)
# y_pred = elastic_net.predict(X_test)

# mse_val = mean_squared_error(y_test, y_pred)
# print("Coefficients:", elastic_net.coef_)
# print("MSE:", mse_val)

# # === Prepare grid for surface ===
# x1_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 30)
# x2_range = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 30)
# x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
# X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))
# y_grid = elastic_net.predict(X_grid)

# # === Create Figure ===
# plt.style.use('seaborn-v0_8-darkgrid')
# fig = plt.figure(figsize=(12, 9))
# ax = fig.add_subplot(111, projection='3d')

# # === Animation Function ===
# def animate(frame):
#     ax.clear()

#     # Titles & labels
#     ax.set_title(f"ElasticNet Regression (α=1.0, l1_ratio=0.5)\nMSE = {mse_val:.2f}",
#                  fontsize=14, fontweight='bold', pad=20)
#     ax.set_xlabel('Feature 1', fontsize=12, labelpad=10)
#     ax.set_ylabel('Feature 2', fontsize=12, labelpad=10)
#     ax.set_zlabel('Target', fontsize=12, labelpad=10)

#     # Rotate camera
#     ax.view_init(elev=20, azim=frame * 2)

#     # Step 1: Actual points
#     if frame >= 0:
#         ax.scatter(X_test[:, 0], X_test[:, 1], y_test,
#                    color='#1f77b4', s=60, alpha=0.9,
#                    edgecolor='black', linewidth=0.5, label='Actual')

#     # Step 2: Predicted points
#     if frame >= 20:
#         ax.scatter(X_test[:, 0], X_test[:, 1], y_pred,
#                    color='#ff7f0e', s=60, alpha=0.9,
#                    edgecolor='black', linewidth=0.5, label='Predicted')

#     # Step 3: Regression surface
#     if frame >= 40:
#         ax.plot_trisurf(x1_grid.flatten(), x2_grid.flatten(), y_grid,
#                         cmap='viridis', alpha=0.5, edgecolor='none')

#     # Legend styling
#     ax.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white')

#     # Adjust limits
#     ax.set_xlim(X_test[:, 0].min(), X_test[:, 0].max())
#     ax.set_ylim(X_test[:, 1].min(), X_test[:, 1].max())

# # === Run Animation ===
# ani = FuncAnimation(fig, animate, frames=100, interval=80, repeat=True)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_regression
# from sklearn.metrics import mean_squared_error
# from mpl_toolkits.mplot3d import Axes3D

# # === Generate regression data ===
# np.random.seed(42)
# X, y = make_regression(n_samples=100, n_features=2, noise=10)
# X = X - np.mean(X, axis=0)  # Centering
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # === Train ElasticNet ===
# elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
# elastic_net.fit(X_train, y_train)
# y_pred = elastic_net.predict(X_test)

# print("Coefficients:", elastic_net.coef_)
# print("MSE:", mean_squared_error(y_test, y_pred))

# # === Create 3D grid for surface ===
# x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
# x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
# x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
# X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))
# y_grid = elastic_net.predict(X_grid)

# # === Setup figure ===
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # === Background colors for smooth transitions ===
# bg_colors = [
#     (0.05, 0.05, 0.1),  # Dark blue
#     (0.1, 0.05, 0.05),  # Dark red
#     (0.05, 0.1, 0.05),  # Dark green
#     (0.08, 0.08, 0.08), # Dark grey
#     (0.05, 0.05, 0.05), # Black
#     (0.1, 0.1, 0.15)    # Blue-grey
# ]

# # === Store plot functions ===
# def plot_regression():
#     ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Actual', alpha=0.8)
#     ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, color='red', label='Predicted', alpha=0.8)
#     ax.plot_trisurf(x1_grid.flatten(), x2_grid.flatten(), y_grid, color='green', alpha=0.5)
#     ax.set_title("ElasticNet Regression", color='white')

# def plot_bar():
#     xpos = np.arange(5)
#     ypos = np.arange(5)
#     xpos, ypos = np.meshgrid(xpos, ypos)
#     xpos = xpos.flatten()
#     ypos = ypos.flatten()
#     zpos = np.zeros_like(xpos)
#     dx = dy = 0.5 * np.ones_like(zpos)
#     dz = np.random.randint(1, 10, size=len(zpos))
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='orange', alpha=0.7)
#     ax.set_title("3D Bar Graph", color='white')

# def plot_histogram():
#     data = np.random.randn(100)
#     hist, bins = np.histogram(data, bins=10)
#     xpos = bins[:-1]
#     ypos = np.zeros_like(xpos)
#     zpos = np.zeros_like(xpos)
#     dx = dy = 0.5 * np.ones_like(zpos)
#     dz = hist
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='purple', alpha=0.7)
#     ax.set_title("3D Histogram", color='white')

# def plot_scatter():
#     x = np.random.rand(50)
#     y_s = np.random.rand(50)
#     z = np.random.rand(50)
#     ax.scatter(x, y_s, z, color='cyan', s=50)
#     ax.set_title("3D Scatter Plot", color='white')

# def plot_area():
#     x = np.linspace(0, 5, 50)
#     y_s = np.linspace(0, 5, 50)
#     x, y_s = np.meshgrid(x, y_s)
#     z = np.sin(x) * np.cos(y_s)
#     ax.plot_surface(x, y_s, z, color='magenta', alpha=0.6)
#     ax.set_title("3D Area Plot", color='white')

# def plot_pie():
#     _theta = np.linspace(0, 2*np.pi, 30)
#     _r = np.linspace(0, 1, 2)
#     T, R = np.meshgrid(_theta, _r)
#     Xp = R * np.cos(T)
#     Yp = R * np.sin(T)
#     Zp = np.zeros_like(Xp)
#     ax.plot_surface(Xp, Yp, Zp, color='gold', alpha=0.7)
#     ax.set_title("3D Pie-like Plot", color='white')

# plots = [plot_regression, plot_bar, plot_histogram, plot_scatter, plot_area, plot_pie]

# # === Animation function ===
# def update(frame):
#     ax.clear()
#     idx = (frame // 60) % len(plots)  # Show each plot for ~2 seconds at 30fps
#     plots[idx]()
#     ax.set_facecolor(bg_colors[idx])
#     ax.view_init(elev=30, azim=frame)  # Smooth rotation
#     return ax,

# # === Create animation ===
# ani = animation.FuncAnimation(fig, update, frames=600, interval=50, blit=False)

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# def get_r(theta,rho):
#   abs_cos=np.abs(np.cos(theta))
#   abs_sin=np.abs(np.sin(theta))
#   a=rho*(abs_cos+abs_sin)
#   b=(1-rho)/2
#   if b==0:
#     r=1/a
#   else:
#     disc=a**2+4*b
#     r=(-a +np.sqrt(disc))/(2*b)
#   return r
# theta=np.linspace(0,2*np.pi,1000)
# r_ridge=get_r(theta,rho=0)
# x_ridge=r_ridge*np.cos(theta)
# y_ridge=r_ridge*np.sin(theta)

# r_lasso=get_r(theta,rho=1)
# x_lasso=r_lasso*np.cos(theta)
# y_lasso=r_lasso*np.sin(theta)

# r_elastic=get_r(theta,rho=0.5)
# x_elastic=r_elastic*np.cos(theta)
# y_elastic=r_elastic*np.sin(theta)

# fig,ax=plt.subplots(figsize=(8,8))
# ax.plot(x_ridge,y_ridge,label='Ridge(L2)',color='red')
# ax.plot(x_lasso,y_lasso,label='Lasso(L1)',color='blue')
# ax.plot(x_elastic,y_elastic,label="elastic(L1+L2)",color='green')
# ax.set_aspect('equal')
# ax.axhline('B1')
# ax.axvline('B2')
# ax.legend()
# ax.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Function to calculate radius for different rho
def get_r(theta, rho):
    abs_cos = np.abs(np.cos(theta))
    abs_sin = np.abs(np.sin(theta))
    a = rho * (abs_cos + abs_sin)
    b = (1 - rho) / 2
    if b == 0:
        r = 1 / a
    else:
        disc = a**2 + 4 * b
        r = (-a + np.sqrt(disc)) / (2 * b)
    return r

# Generate mesh for 3D extrusion
theta = np.linspace(0, 2*np.pi, 200)
z = np.linspace(-1, 1, 50)
Theta, Z = np.meshgrid(theta, z)

# Ridge
r_ridge = get_r(Theta, rho=0)
X_ridge = r_ridge * np.cos(Theta)
Y_ridge = r_ridge * np.sin(Theta)

# Lasso
r_lasso = get_r(Theta, rho=1)
X_lasso = r_lasso * np.cos(Theta)
Y_lasso = r_lasso * np.sin(Theta)

# ElasticNet
r_elastic = get_r(Theta, rho=0.5)
X_elastic = r_elastic * np.cos(Theta)
Y_elastic = r_elastic * np.sin(Theta)

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surfaces
surf_ridge = ax.plot_surface(X_ridge, Y_ridge, Z, color='red', alpha=0.5)
surf_lasso = ax.plot_surface(X_lasso, Y_lasso, Z, color='blue', alpha=0.5)
surf_elastic = ax.plot_surface(X_elastic, Y_elastic, Z, color='green', alpha=0.5)

# Axis labels and title
ax.set_xlabel('B1', fontsize=12)
ax.set_ylabel('B2', fontsize=12)
ax.set_zlabel('Height', fontsize=12)
ax.set_title('3D Regularization Constraints', fontsize=14)

# Legend substitute
ax.text(2.2, 0, 1.0, 'Ridge (L2)', color='red', fontsize=10)
ax.text(0, 2.2, 1.0, 'Lasso (L1)', color='blue', fontsize=10)
ax.text(-2.2, 0, 1.0, 'ElasticNet', color='green', fontsize=10)

# Animation function
def update(frame):
    ax.view_init(elev=20, azim=frame)
    return fig,

# Create animation (slow rotation for clarity)
anim = FuncAnimation(fig, update, frames=np.linspace(0, 360, 180), interval=80, blit=False)

plt.show()
