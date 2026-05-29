import os
import zipfile
import urllib.request

def download_and_extract():
    url = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
    zip_path = "realwaste.zip"
    extract_dir = "realwaste_dataset"

    print(f"Downloading RealWaste dataset from: {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    print(f"Extracting dataset to '{extract_dir}'...")
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete!")
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return
    finally:
        # Clean up zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up temporary zip file.")

    print("\nDataset is ready!")
    print(f"Files are located in: {os.path.abspath(extract_dir)}")

if __name__ == "__main__":
    download_and_extract()
