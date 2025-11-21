import subprocess
import sys
import os

SCRIPT = os.path.join("src", "query_interface.py")

def main():
    print("\nLaunching Movie Sentiment Recommender...\n")
    subprocess.run([sys.executable, SCRIPT])


if __name__ == "__main__":
    main()