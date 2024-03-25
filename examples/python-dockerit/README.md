# DockerIt

DockerIt is a tool to help you build and run your application in a Docker container. It consists of a model that defines the system prompt and model weights to use, along with a python script to then build the container and run the image automatically.

## Running the Example

1. Ensure you have the `mattw/dockerit` model installed:

   ```bash
   ollama pull mattw/dockerit
   ```

2. Make sure Docker is running on your machine.

3. Install the Python Requirements.

   ```bash
   pip install -r requirements.txt
   ```

4. Run the example:

   ```bash
   python dockerit.py "simple postgres server with admin password set to 123"
   ```

5. Enter the name you would like to use for your container image.

## Caveats

This is a simple example. It's assuming the Dockerfile content generated is going to work. In many cases, even with simple web servers, it fails when trying to copy files that don't exist. It's simply an example of what you could possibly do.
