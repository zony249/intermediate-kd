import os 
import sys 


import wget

def mkdir(dir: str):
    os.makedirs(dir, exist_ok=True)


def download(source_url:str, dest_path:str):
    if not os.path.exists(dest_path):
        print(f"downloading {source_url}...")
        wget.download(source_url, out=dest_path)
    else:
        print(f"{dest_path} is already downloaded")

if __name__ == "__main__":
    pass
