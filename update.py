import subprocess

fetch_output = subprocess.Popen('git fetch origin', shell=True, stdout=subprocess.PIPE).communicate()[0].decode('utf-8').strip()
merge_output = subprocess.Popen('git merge origin/main', shell=True, stdout=subprocess.PIPE).communicate()[0].decode('utf-8').strip()
print('fetch output:')
print(fetch_output)
print('merge output:')
print(merge_output)
