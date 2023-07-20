# progressbar

[![CI](https://github.com/schollz/progressbar/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/schollz/progressbar/actions/workflows/ci.yml)
[![go report card](https://goreportcard.com/badge/github.com/schollz/progressbar)](https://goreportcard.com/report/github.com/schollz/progressbar) 
[![coverage](https://img.shields.io/badge/coverage-84%25-brightgreen.svg)](https://gocover.io/github.com/schollz/progressbar)
[![godocs](https://godoc.org/github.com/schollz/progressbar?status.svg)](https://godoc.org/github.com/schollz/progressbar/v3) 

A very simple thread-safe progress bar which should work on every OS without problems. I needed a progressbar for [croc](https://github.com/schollz/croc) and everything I tried had problems, so I made another one. In order to be OS agnostic I do not plan to support [multi-line outputs](https://github.com/schollz/progressbar/issues/6).


## Install

```
go get -u github.com/schollz/progressbar/v3
```

## Usage 

### Basic usage

```golang
bar := progressbar.Default(100)
for i := 0; i < 100; i++ {
    bar.Add(1)
    time.Sleep(40 * time.Millisecond)
}
```

which looks like:

![Example of basic bar](examples/basic/basic.gif)


### I/O operations

The `progressbar` implements an `io.Writer` so it can automatically detect the number of bytes written to a stream, so you can use it as a progressbar for an `io.Reader`.

```golang
req, _ := http.NewRequest("GET", "https://dl.google.com/go/go1.14.2.src.tar.gz", nil)
resp, _ := http.DefaultClient.Do(req)
defer resp.Body.Close()

f, _ := os.OpenFile("go1.14.2.src.tar.gz", os.O_CREATE|os.O_WRONLY, 0644)
defer f.Close()

bar := progressbar.DefaultBytes(
    resp.ContentLength,
    "downloading",
)
io.Copy(io.MultiWriter(f, bar), resp.Body)
```

which looks like:

![Example of download bar](examples/download/download.gif)


### Progress bar with unknown length

A progressbar with unknown length is a spinner. Any bar with -1 length will automatically convert it to a spinner with a customizable spinner type. For example, the above code can be run and set the `resp.ContentLength` to `-1`.

which looks like:

![Example of download bar with unknown length](examples/download-unknown/download-unknown.gif)


### Customization

There is a lot of customization that you can do - change the writer, the color, the width, description, theme, etc. See [all the options](https://pkg.go.dev/github.com/schollz/progressbar/v3?tab=doc#Option).

```golang
bar := progressbar.NewOptions(1000,
    progressbar.OptionSetWriter(ansi.NewAnsiStdout()),
    progressbar.OptionEnableColorCodes(true),
    progressbar.OptionShowBytes(true),
    progressbar.OptionSetWidth(15),
    progressbar.OptionSetDescription("[cyan][1/3][reset] Writing moshable file..."),
    progressbar.OptionSetTheme(progressbar.Theme{
        Saucer:        "[green]=[reset]",
        SaucerHead:    "[green]>[reset]",
        SaucerPadding: " ",
        BarStart:      "[",
        BarEnd:        "]",
    }))
for i := 0; i < 1000; i++ {
    bar.Add(1)
    time.Sleep(5 * time.Millisecond)
}
```

which looks like:

![Example of customized bar](examples/customization/customization.gif)


## Contributing

Pull requests are welcome. Feel free to...

- Revise documentation
- Add new features
- Fix bugs
- Suggest improvements

## Thanks

Thanks [@Dynom](https://github.com/dynom) for massive improvements in version 2.0!

Thanks [@CrushedPixel](https://github.com/CrushedPixel) for adding descriptions and color code support!

Thanks [@MrMe42](https://github.com/MrMe42) for adding some minor features!

Thanks [@tehstun](https://github.com/tehstun) for some great PRs!

Thanks [@Benzammour](https://github.com/Benzammour) and [@haseth](https://github.com/haseth) for helping create v3!

Thanks [@briandowns](https://github.com/briandowns) for compiling the list of spinners.

## License

MIT
