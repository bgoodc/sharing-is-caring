import tarfile
import sys
import os
from urllib.request import urlretrieve

download_dir = '.'
download_tmp = os.path.join(download_dir, 'tmp.tar.gz')
download_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

if not os.path.exists(download_dir):
    os.mkdir(download_dir)

urlretrieve(download_url, download_tmp)
tar = tarfile.open(download_tmp)
tar.extractall(download_dir)
os.remove(download_tmp)
