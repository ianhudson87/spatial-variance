'''
Put tarball in ./data
Run this
'''
import os
import tarfile

data_file_name = "knee_singlecoil_test_v2.tar"
data_path = os.path.join("data", data_file_name)

file_content = tarfile.open(data_path, 'r')
file_content.extractall(os.path.join("data"))
print(file_content.list())
file_content.close()