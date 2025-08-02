# DockerIt

DockerIt is a tool to help you build and run your application in a Docker container. It consists of a model that defines the system prompt and model weights to use, along with a python script to then build the container and run the image automatically. 

## Caveats

This is an simple example. It's assuming the Dockerfile content generated is going to work. In many cases, even with simple web servers, it fails when trying to copy files that don't exist. It's simply an example of what you could possibly do.

## Example Usage

```bash
> python3 ./dockerit.py "simple postgres server with admin password set to 123"
Enter the name of the image: matttest
Container named happy_keller  started with id:  7c201bb6c30f02b356ddbc8e2a5af9d7d7d7b8c228519c9a501d15c0bd9d6b3e
```
