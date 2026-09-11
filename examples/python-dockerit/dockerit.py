import requests, json, docker, io, sys
inputDescription = " ".join(sys.argv[1:])
imageName = input("Enter the name of the image: ")
client = docker.from_env()
s = requests.Session()
output=""
with s.post('http://localhost:11434/api/generate', json={'model': 'mattw/dockerit', 'prompt': inputDescription}, stream=True) as r:
  for line in r.iter_lines():
    if line:
      j = json.loads(line)
      if "response" in j:
        output = output +j["response"]
output = output[output.find("---start")+9:output.find("---end")-1]
f = io.BytesIO(bytes(output, 'utf-8'))
client.images.build(fileobj=f, tag=imageName)
container = client.containers.run(imageName, detach=True)
print("Container named", container.name, " started with id: ",container.id)
