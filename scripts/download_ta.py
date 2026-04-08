import urllib.request
import tarfile
import os
import shutil

def download_ta():
    url = "https://files.pythonhosted.org/packages/source/t/ta/ta-0.11.0.tar.gz"
    filename = "ta-0.11.0.tar.gz"
    
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filename)
    
    print("Extracting...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
    
    src = os.path.join("ta-0.11.0", "ta")
    dest = os.path.join("public", "ta")
    
    if os.path.exists(dest):
        shutil.rmtree(dest)
        
    print(f"Moving {src} to {dest}...")
    shutil.copytree(src, dest)
    
    # Cleanup
    os.remove(filename)
    shutil.rmtree("ta-0.11.0")
    print("Done!")

if __name__ == "__main__":
    download_ta()
