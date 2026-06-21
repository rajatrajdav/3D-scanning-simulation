import requests
import json
import zipfile
import os
import sys

# --- Configuration ---
API_KEY = "kiri_fY_P0PneyULXY8u0TDrf-Lf6o5iy0-1DgkK9Ad430rs"
ENDPOINT = "https://api.kiriengine.app/api/v1/open/model/getModelZip"

def fetch_and_extract_model(serialize_id="5e82c7d449ae4754a2b140c014624f87"):
    # Use command-line argument or default
    if serialize_id is None:
        if len(sys.argv) > 1:
            serialize_id = sys.argv[1]
        else:
            print("Error: Please provide a serialize ID")
            print("Usage: python download.py <serialize_id>")
            return
    
    print(f"Requesting download URL for task: {serialize_id}...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    params = {
        "serialize": serialize_id
    }

    try:
        # 1. Request the model URL from the Kiri Engine API
        response = requests.get(ENDPOINT, headers=headers, params=params)
        
        if response.status_code == 200:
            res_data = response.json()
            
            # Extract modelUrl from the nested data dictionary
            model_url = res_data.get("data", {}).get("modelUrl")
            
            if model_url == "xxx":
                print("\n[Status] The model is still being generated on the cloud servers.")
                print("Please wait 2-3 minutes and run this script again.")
                return
                
            if model_url and model_url.startswith("http"):
                zip_filename = f"{serialize_id}.zip"
                extract_folder = f"./model_{serialize_id}"
                
                # 2. Download the ZIP file stream
                print(f"Link obtained! Downloading archive to {zip_filename}...")
                with requests.get(model_url, stream=True) as file_stream:
                    file_stream.raise_for_status()
                    with open(zip_filename, 'wb') as f:
                        for chunk in file_stream.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                # 3. Unpack the ZIP archive to extract your STL/OBJ file
                print(f"Download complete. Extracting files to {extract_folder}...")
                os.makedirs(extract_folder, exist_ok=True)
                with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)
                
                # Clean up the zip file after extraction
                os.remove(zip_filename)
                print(f"\n[Success] Your 3D files are ready in: {os.path.abspath(extract_folder)}")
                
            else:
                print("Error: Could not find a valid download link in the response data.")
                print(json.dumps(res_data, indent=2))
        else:
            print(f"API Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    fetch_and_extract_model()