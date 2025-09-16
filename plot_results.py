import numpy as np
import matplotlib.pyplot as plt

def parse_results_file(filename):
    """
    Reads a results file with multiple trials and returns the labeled samples,
    accuracies, and query times in a structured format.
    """
    all_trials_accuracies = []
    all_trials_times = []
    current_trial_accuracies = []
    current_trial_times = []
    x_axis_samples = []

    try:
        with open(filename, 'r') as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        print(f"Warning: The file '{filename}' was not found. Skipping.")
        return None, None, None

    for line in all_lines:
        line = line.strip()

        if line.startswith('---') or not line:
            if current_trial_accuracies:
                all_trials_accuracies.append(current_trial_accuracies)
                all_trials_times.append(current_trial_times)
            current_trial_accuracies = []
            current_trial_times = []
            continue

        parts = line.split(", ")
        sample_str = parts[1].split(": ")[1]
        accuracy_str = parts[2].split(": ")[1].strip('%\n')
        # --- MODIFICATION: Parse the Query Time ---
        time_str = parts[3].split(": ")[1].strip('s\n')

        sample_count = int(sample_str)
        accuracy = float(accuracy_str)
        query_time = float(time_str)

        current_trial_accuracies.append(accuracy)
        current_trial_times.append(query_time)

        if len(all_trials_accuracies) == 0:
            x_axis_samples.append(sample_count)

    if current_trial_accuracies:
        all_trials_accuracies.append(current_trial_accuracies)
        all_trials_times.append(current_trial_times)

    return x_axis_samples, np.array(all_trials_accuracies), np.array(all_trials_times)

# --- 1. Parse Data for All Four Experiments ---
x_axis, badge_acc, badge_time = parse_results_file("results_badge.txt")
_, random_acc, random_time = parse_results_file("results_random.txt")
_, subset_acc, subset_time = parse_results_file("results_subset_sampling.txt")
_, proxy_acc, proxy_time = parse_results_file("results_proxy.txt")


# --- 2. Create the Accuracy Plot ---
plt.figure(figsize=(10, 7))

# Only average and plot the data if it was successfully loaded
if badge_acc is not None:
    avg_badge_acc = np.mean(badge_acc, axis=0)
    plt.plot(x_axis, avg_badge_acc, marker='o', linestyle='-', label='Full BADGE')

if subset_acc is not None:
    avg_subset_acc = np.mean(subset_acc, axis=0)
    plt.plot(x_axis, avg_subset_acc, marker='^', linestyle='-.', label='Subset Sampling BADGE')

if proxy_acc is not None:
    avg_proxy_acc = np.mean(proxy_acc, axis=0)
    plt.plot(x_axis, avg_proxy_acc, marker='x', linestyle=':', label='Proxy-Based BADGE')

if random_acc is not None:
    avg_random_acc = np.mean(random_acc, axis=0)
    plt.plot(x_axis, avg_random_acc, marker='s', linestyle='--', label='Random Sampling')

plt.title('Algorithm Performance Comparison (5 Trial Average)')
plt.xlabel('Number of Labeled Samples')
plt.ylabel('Average Test Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_comparison.png')
print("Accuracy comparison plot saved as accuracy_comparison.png")
plt.show()


# --- 3. Create the Query Time Plot ---
plt.figure(figsize=(10, 7))

# Only average and plot the data if it was successfully loaded
if badge_time is not None:
    avg_badge_time = np.mean(badge_time, axis=0)
    plt.plot(x_axis, avg_badge_time, marker='o', linestyle='-', label='Full BADGE')

if subset_time is not None:
    avg_subset_time = np.mean(subset_time, axis=0)
    plt.plot(x_axis, avg_subset_time, marker='^', linestyle='-.', label='Subset Sampling BADGE')

if proxy_time is not None:
    avg_proxy_time = np.mean(proxy_time, axis=0)
    plt.plot(x_axis, avg_proxy_time, marker='x', linestyle=':', label='Proxy-Based BADGE')

if random_time is not None:
    avg_random_time = np.mean(random_time, axis=0)
    plt.plot(x_axis, avg_random_time, marker='s', linestyle='--', label='Random Sampling')

plt.title('Query Time Comparison (5 Trial Average)')
plt.xlabel('Number of Labeled Samples')
plt.ylabel('Average Query Time (seconds)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('query_time_comparison.png')
print("Query time comparison plot saved as query_time_comparison.png")
plt.show()