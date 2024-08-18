# a really really simple web client

This is a web Client with almost NO features but a small starting point for
people who want to try for themselves.
It utilizes a simple XmlHttpRequest and prints the output to screen.

I also implemented the use of marked for highlighting the chat.

Since there is no on-the-fly-reception, getting an answer may take a while...

# usage

## using it locally

if you just want to try, run firefox in your command line like this:

```
$ firefox file:///path/to/ollama/examples/simple-webclient/webcli.html?host=your_hostname
```

This opens your browser (eg firefox) and in this case, directly sets ollama host to
_http://your_hostname:11434_.
Default host is either the host where the script runs or just `localhost`.

For more configuration, see `Configuring`

## using behind nginx

The most comfortable way I found was using the cli on nginx.

! This is only an example. You should use a dedicated site in nginx !

Therefore, I just copied webcli.html to /var/www/html/.

To make it run, you might need to edit the location in `/etc/nginx/sites-available/default`
and add

```
...
    location / {
        ...
        add_header 'Access-Control-Allow-Origin' '*';
        ...
    }
...
```

After that, you need to reload nginx.

Now, you should be able to access webcli via `http://your_host/webcli.html`


## Configuring

If you want to configure a bit more, just click the "Configure" link below your chat input.
A form opens and you can input hostname, port, whether using https, as well as the used
model, parameters and system input.

# Todo

Well, there's still something to do here. Source code formatting would be cool or maybe
saving the configuration somehow.. Feel free..
