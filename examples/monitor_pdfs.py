import os
import time
import subprocess
import sys


def scan_pdfs(root_dir):
    pdfs = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.pdf') and "deformed" in filename.lower():
                full_path = os.path.join(dirpath, filename)
                pdfs.add(full_path)
    return pdfs


def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python monitor_pdfs.py /path/to/directory")
    #     sys.exit(1)
    # root_dir = sys.argv[1]

    # if not os.path.isdir(root_dir):
    #     print(f"The path {root_dir} is not a valid directory.")
    #     sys.exit(1)

    root_dir = r"/home/fryderyk/Documents/code/registrationbaselines/tmp/displacement_debug/LungCT/BSplineNiftyReg"

    existing_pdfs = scan_pdfs(root_dir)
    print(f"Monitoring '{root_dir}' for new PDFs...")
    while True:
        time.sleep(0.1)
        current_pdfs = scan_pdfs(root_dir)
        new_pdfs = current_pdfs - existing_pdfs
        if new_pdfs:
            for pdf in new_pdfs:
                print(f"New PDF detected: {pdf}")
                # Open in Visual Studio Code
                try:
                    subprocess.Popen(args=['xdg-open', pdf])
                    # subprocess.Popen(['code', pdf])
                except Exception as e:
                    print(f"Error opening {pdf} in Visual Studio Code: {e}")
            existing_pdfs.update(new_pdfs)


if __name__ == "__main__":
    main()
