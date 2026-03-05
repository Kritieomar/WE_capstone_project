import requests

base_url = "http://localhost:8000/api"

# 1. Upload files
print("Uploading files...")
with open("test_model.joblib", "rb") as mf:
    with open("test_dataset.csv", "rb") as df:
        files = {
            "model_file": mf,
            "data_file": df
        }
        data = {
            "target_col": "target"
        }
        r = requests.post(f"{base_url}/upload", files=files, data=data)
        print("Upload Response:", r.status_code, r.text)

# 2. Global Explain
print("\nRequesting global explanation...")
r = requests.get(f"{base_url}/explain/global")
print("Global Explain Status:", r.status_code)
try:
    print(r.json())
except Exception as e:
    print("Cannot parse JSON:", r.text)

