# DevOps Engineer Modelfile

This is a simple modelfile that takes will respond to any questions that a senior DevOps engineer should know how to solve.

````text
ollama create devopsengineer

ollama run devopsengineer
>>> how can i proxy all connections from inside the office through a single server that caches large files. Provide a config file for nginx to achieve this?

Here is an example configuration file for Nginx that can be used to proxy all connections from inside the office network to a centralized cache server:
```perl
http {
    upstream cache_server {
        server <cache_server_ip>;
        server <cache_server_ip>;
        server <cache_server_ip>;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://cache_server;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
```
In this configuration, we define an `upstream` block that lists the IP addresses of the cache servers. We then define a server block that listens on port 80 and routes all 
traffic to the `/` location to the cache servers using the `proxy_pass` directive. The `proxy_set_header` directives are used to preserve the source IP address of the client
request when forwarding it to the cache server.

To use this configuration, you would need to replace the placeholder `<cache_server_ip>` with the actual IP addresses of your cache servers. You would also need to make sure
that the cache servers are configured to accept incoming connections from the Nginx server and handle requests for files.

````