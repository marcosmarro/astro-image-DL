import subprocess
from utils import plot_comparisons, plot_cross_correlation

models = ['original', 'n2v', 'calibrated', 'n2n']

# Train on calibrated data
print("Training N2V model on calibrated data...")
subprocess.run("python train.py -d Training -m n2v", check=True, shell=True)

print("Training N2N model on calibrated data...")
subprocess.run("python train.py -d Training -m n2n", check=True, shell=True)


# Inference on calibrated data
print("Performing N2V inference on calibrated data...")
subprocess.run("python inference.py -d Science -m n2v", check=True, shell=True)

print("Performing N2N inference on calibrated data...")
subprocess.run("python inference.py -d Science -m n2n", check=True, shell=True)


# Evaluation on calibrated data
print("Evaluating N2V results on calibrated data...")
subprocess.run("python evaluation.py -d Denoised_Science -m n2v", check=True, shell=True)

print("Evaluating N2N results on calibrated data...")
subprocess.run("python evaluation.py -d Denoised_Science -m n2n", check=True, shell=True)


# Plotting
# print("Generating plots...")
# plot_comparisons(models)
# plot_cross_correlation(models)

# Done
print("Done!")
