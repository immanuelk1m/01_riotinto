# Runs in the Colab kernel: files to download for direction __DIR__.
import glob
for f in sorted(glob.glob("/content/E008/results/__DIR__/*") + glob.glob("/content/E008/logs/*__DIR__*")):
    print(f)
