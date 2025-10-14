import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_error_decay_curve():
    # Create time data (x-axis)
    t = np.linspace(0, 10, 200)  # time from 0 to 10 seconds

    # Define the curve parameters
    a = 0.8   # initial height of the curve
    b = 0.5   # decay rate (larger = faster decay)
    c = 0.2   # final plateau value (non-zero)

    # Compute the error over time: e(t)
    e = a * np.exp(-b * t) + c

    # Plot the curve
    plt.figure(figsize=(8, 5))
    plt.plot(t, e, color='royalblue', linewidth=2)
    plt.title("Error Decay Over Time", fontsize=14)
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Error (e)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot as a PNG file
    plt.savefig("pid_error_decay_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Graph saved as 'pid_error_decay_curve.png'")

def plot_scatter_plot():
    # Step 1: Create the dummy data
    data = {
        'Temperature (°C)': [20, 22, 25, 28, 30],
        'Coffee Sales ($)': [250, 280, 320, 370, 410]
    }

    # Step 2: Convert to DataFrame
    df = pd.DataFrame(data)

    # Step 3: Plot scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(df['Temperature (°C)'], df['Coffee Sales ($)'], color='brown', marker='o')

    # Step 4: Customize the plot
    plt.title('Coffee Sales vs Temperature')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Coffee Sales ($)')
    plt.grid(True)
    plt.tight_layout()

    # Step 5: Show the plot
    # Save the plot as a PNG file
    plt.savefig("coffee_sales_vs_temp.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Graph saved as 'coffee_sales_vs_temp.png'")

def plot_regression_line():
    # Create the dummy data
    data = {
        'Temperature (°C)': [20, 22, 25, 28, 30],
        'Coffee Sales ($)': [250, 280, 320, 370, 410]
    }

    df = pd.DataFrame(data)

    # Extract x and y values
    x = df['Temperature (°C)']
    y = df['Coffee Sales ($)']

    # Perform linear regression (degree 1 polynomial)
    coefficients = np.polyfit(x, y, deg=1)
    slope, intercept = coefficients

    # Generate predicted y values for the best-fit line
    y_pred = slope * x + intercept

    # Plot the scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='brown', label='Actual Data')

    # Plot the best-fit line
    plt.plot(x, y_pred, color='blue', linestyle='--', label=f'Best Fit Line: y = {slope:.2f}x + {intercept:.2f}')

    # Customize the plot
    plt.title('Coffee Sales vs Temperature with Regression Line')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Coffee Sales ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.savefig("coffee_sales_vs_temp_with_regression_line.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Graph saved as 'coffee_sales_vs_temp_with_regression_line.png'")

if __name__ == "__main__":
    # plot_error_decay_curve()
    plot_scatter_plot()
    plot_regression_line()