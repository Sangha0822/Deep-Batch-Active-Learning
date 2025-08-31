import numpy as np

def parse_results_file(filename):
    all_trials_accuracies = []
    current_trial_accuracies = []
    x_axis_samples = [] 

    try:
        with open(filename, 'r') as open_file:
            all_lines = open_file.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find the file '{filename}'")
        return None, None

    for line in all_lines:
        line = line.strip()

        if line.startswith('---') or not line:
            if current_trial_accuracies:
                all_trials_accuracies.append(current_trial_accuracies)
            current_trial_accuracies = []
            continue

        parts = line.split(", ")
        sample_str = parts[1].split(": ")[1]
        accuracy_str = parts[2].split(": ")[1].strip('%\n')

        sample_count = int(sample_str)
        accuracy = float(accuracy_str)

        current_trial_accuracies.append(accuracy)

        if len(all_trials_accuracies) == 0:
            x_axis_samples.append(sample_count)

    if current_trial_accuracies:
        all_trials_accuracies.append(current_trial_accuracies)

    return x_axis_samples, all_trials_accuracies



x_axis, badge_data = parse_results_file("results_badge.txt")

_, random_data = parse_results_file("results_random.txt")


if badge_data and random_data:
    print("--- BADGE Data (First Trial) ---")
    print(badge_data[0])
    
    print("\n--- Random Sampling Data (First Trial) ---")
    print(random_data[0])


badge_data_np = np.array(badge_data)
random_data_np = np.array(random_data)

avg_badge_accuracies = np.mean(badge_data_np, axis=0)
avg_random_accuracies = np.mean(random_data_np, axis=0)

print("Final Averaged BADGE Accuracies:", avg_badge_accuracies)
print("Final Averaged Random Accuracies:", avg_random_accuracies)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7)) 

plt.plot(x_axis, avg_badge_accuracies, marker='o', linestyle='-', label='BADGE Algorithm')


plt.plot(x_axis, avg_random_accuracies, marker='s', linestyle='--', label='Random Sampling')


plt.title('BADGE vs. Random Sampling Performance on MNIST (5 Trial Average)')
plt.xlabel('Number of Labeled Samples')
plt.ylabel('Average Test Accuracy (%)')
plt.legend() # Displays the labels for each curve
plt.grid(True) # Adds a grid for easier reading


plt.savefig('badge_vs_random_comparison.png')

plt.show()