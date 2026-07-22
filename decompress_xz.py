import os
import glob
import lzma
import shutil
import time

def decompress_folder(folder_path):
    print(f"Decompressing files in {folder_path}...")
    xz_files = glob.glob(os.path.join(folder_path, "*.xz"))
    total = len(xz_files)
    
    for idx, xz_path in enumerate(xz_files, 1):
        # Target path: remove .xz extension
        nc_path = xz_path[:-3]
        if os.path.exists(nc_path):
            print(f"[{idx}/{total}] Already exists: {os.path.basename(nc_path)}")
            continue
            
        print(f"[{idx}/{total}] Decompressing {os.path.basename(xz_path)} -> {os.path.basename(nc_path)}...")
        t0 = time.time()
        try:
            with lzma.open(xz_path, 'rb') as f_in:
                with open(nc_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"    Done in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"    Error decompressing {xz_path}: {e}")

if __name__ == "__main__":
    decompress_folder("../ocean_tide_extrapolated")
    decompress_folder("../load_tide")
