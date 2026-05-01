# DVC setup — run these commands to initialize

# 1. Initialize DVC
dvc init

# 2. Track raw dataset
dvc add data/raw/irrigation_data.csv

# 3. Commit DVC metadata
git add data/raw/irrigation_data.csv.dvc .dvc/
git commit -m "track dataset with DVC"

# 4. Add remote (example: local or S3)
dvc remote add -d myremote s3://my-bucket/irrigation
# or for local:
# dvc remote add -d myremote /tmp/dvc-remote

# 5. Push data
dvc push
