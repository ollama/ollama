# Running Ollama on Fly.io GPU Instances

Ollama runs with little to no configuration on [Fly.io GPU instances](https://fly.io/docs/gpus/gpu-quickstart/). If you don't have access to GPUs yet, you'll need to [apply for access](https://fly.io/gpu/) on the waitlist. Once you're accepted, you'll get an email with instructions on how to get started.

Create a new app with `fly apps create`:

```bash
fly apps create
```

Then create a `fly.toml` file in a new folder that looks like this:

```toml
app = "sparkling-violet-709"
primary_region = "ord"
vm.size = "a100-40gb" # see https://fly.io/docs/gpus/gpu-quickstart/ for more info

[build]
  image = "ollama/ollama"

[http_service]
  internal_port = 11434
  force_https = false
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
  processes = ["app"]

[mounts]
  source = "models"
  destination = "/root/.ollama"
  initial_size = "100gb"
```

Then create a [new private IPv6 address](https://fly.io/docs/reference/private-networking/#flycast-private-load-balancing) for your app:

```bash
fly ips allocate-v6 --private
```

Then deploy your app:

```bash
fly deploy
```

And finally you can access it interactively with a new Fly.io Machine:

```
fly machine run -e OLLAMA_HOST=http://your-app-name.flycast --shell ollama/ollama
```

```bash
$ ollama run openchat:7b-v3.5-fp16
>>> How do I bake chocolate chip cookies?
 To bake chocolate chip cookies, follow these steps:

1. Preheat the oven to 375°F (190°C) and line a baking sheet with parchment paper or silicone baking mat.

2. In a large bowl, mix together 1 cup of unsalted butter (softened), 3/4 cup granulated sugar, and 3/4
cup packed brown sugar until light and fluffy.

3. Add 2 large eggs, one at a time, to the butter mixture, beating well after each addition. Stir in 1
teaspoon of pure vanilla extract.

4. In a separate bowl, whisk together 2 cups all-purpose flour, 1/2 teaspoon baking soda, and 1/2 teaspoon
salt. Gradually add the dry ingredients to the wet ingredients, stirring until just combined.

5. Fold in 2 cups of chocolate chips (or chunks) into the dough.

6. Drop rounded tablespoons of dough onto the prepared baking sheet, spacing them about 2 inches apart.

7. Bake for 10-12 minutes, or until the edges are golden brown. The centers should still be slightly soft.

8. Allow the cookies to cool on the baking sheet for a few minutes before transferring them to a wire rack
to cool completely.

Enjoy your homemade chocolate chip cookies!
```

When you set it up like this, it will automatically turn off when you're done using it. Then when you access it again, it will automatically turn back on. This is a great way to save money on GPU instances when you're not using them. If you want a persistent wake-on-use connection to your Ollama instance, you can set up a [connection to your Fly network using WireGuard](https://fly.io/docs/reference/private-networking/#discovering-apps-through-dns-on-a-wireguard-connection). Then you can access your Ollama instance at `http://your-app-name.flycast`.

And that's it!
