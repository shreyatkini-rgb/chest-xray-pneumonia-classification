import kagglehub
import shutil
import os

print("Downloading dataset...")

path = kagglehub.dataset_download(
    "paultimothymooney/chest-xray-pneumonia"
)

print("Dataset downloaded to:")
print(path)

destination = os.path.join("data", "chest_xray")

if os.path.exists(destination):
    print("Destination already exists:", destination)
else:
    os.makedirs("data", exist_ok=True)
    shutil.copytree(path, destination)
    print("Dataset copied to:", destination)

print("\nDataset setup complete!")