import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import math 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example data: replace with your actual results
turns = np.array([5, 10, 15, 20, 25])
accuracy_data = {
    'TOP-1': np.array([0.32, 0.44, 0.4333, 0.3933, 0.4533]),
    'TOP-3': np.array([0.3867, 0.4933, 0.4933, 0.54, 0.5133]),
    'TOP-5': np.array([0.4733, 0.5133, 0.54, 0.5667, 0.5133]),
    'TOP-7': np.array([0.5467, 0.5867, 0.5667, 0.6067, 0.5467])
}
def compute_sigma(x, y, mu):
    """Compute standard deviation based on spread around mu."""
    return np.sqrt(np.sum(y * (x - mu)**2) / np.sum(y))

def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-((x - mu)**2) / (2 * sigma**2))

colors = ['#f4ced2', '#c98882', '#a33937', '#662019']  # hex colors
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']

plt.figure(figsize=(10,6))
x_smooth = np.linspace(turns.min(), turns.max(), 200)

for i, (key, acc) in enumerate(accuracy_data.items()):
    mu = turns[np.argmax(acc)]  # turn with maximum accuracy
    sigma = compute_sigma(turns, acc, mu)
    plt.plot(x_smooth, gaussian(x_smooth, acc.max(), mu, sigma),
             label=f'{key} μ={mu}', color=colors[i],
             linestyle=linestyles[i], alpha=0.7, linewidth=4)
    plt.scatter(turns, acc, color=colors[i])

plt.xticks(turns)  # only show your original turn values
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
plt.xlabel('Number of Turns')
plt.ylabel('Accuracy')
plt.title('Gaussian Bell Curves of Accuracy vs Turns per TOP-K')
plt.legend(title='TOP-K (μ = average accuracy)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print computed μ and σ for reference
for key, acc in accuracy_data.items():
    mu = turns[np.argmax(acc)]
    sigma = compute_sigma(turns, acc, mu)
    print(f"{key}: μ = {mu}, σ = {sigma:.2f}")