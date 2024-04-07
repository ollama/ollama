package cmd

import (
	"archive/zip"
	"bytes"
	"errors"
	"fmt"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/spf13/cobra"
	"io"
	"os"
	"path/filepath"
	"strings"
)

var createCmd = &cobra.Command{
	Use:     "create MODEL",
	Short:   "Create a model from a Modelfile",
	Args:    cobra.ExactArgs(1),
	PreRunE: checkServerHeartbeat,
	RunE:    CreateHandler,
}

func init() {
	createCmd.Flags().StringP("file", "f", "Modelfile", "Name of the Modelfile (default \"Modelfile\")")
	appendHostEnvDocs(createCmd)
}

func CreateHandler(cmd *cobra.Command, args []string) error {
	filename, _ := cmd.Flags().GetString("file")
	filename, err := filepath.Abs(filename)
	if err != nil {
		return err
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)

	modelfile, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	commands, err := parser.Parse(bytes.NewReader(modelfile))
	if err != nil {
		return err
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	status := "transferring model data"
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)

	for _, c := range commands {
		switch c.Name {
		case "model", "adapter":
			path := c.Args
			if path == "~" {
				path = home
			} else if strings.HasPrefix(path, "~/") {
				path = filepath.Join(home, path[2:])
			}

			if !filepath.IsAbs(path) {
				path = filepath.Join(filepath.Dir(filename), path)
			}

			fi, err := os.Stat(path)
			if errors.Is(err, os.ErrNotExist) && c.Name == "model" {
				continue
			} else if err != nil {
				return err
			}

			// TODO make this work w/ adapters
			if fi.IsDir() {
				tf, err := os.CreateTemp("", "ollama-tf")
				if err != nil {
					return err
				}
				defer os.RemoveAll(tf.Name())

				zf := zip.NewWriter(tf)

				files, err := filepath.Glob(filepath.Join(path, "model-*.safetensors"))
				if err != nil {
					return err
				}

				if len(files) == 0 {
					return fmt.Errorf("no safetensors files were found in '%s'", path)
				}

				// add the safetensor config file + tokenizer
				files = append(files, filepath.Join(path, "config.json"))
				files = append(files, filepath.Join(path, "added_tokens.json"))
				files = append(files, filepath.Join(path, "tokenizer.model"))

				for _, fn := range files {
					f, err := os.Open(fn)
					if os.IsNotExist(err) && strings.HasSuffix(fn, "added_tokens.json") {
						continue
					} else if err != nil {
						return err
					}

					fi, err := f.Stat()
					if err != nil {
						return err
					}

					h, err := zip.FileInfoHeader(fi)
					if err != nil {
						return err
					}

					h.Name = filepath.Base(fn)
					h.Method = zip.Store

					w, err := zf.CreateHeader(h)
					if err != nil {
						return err
					}

					_, err = io.Copy(w, f)
					if err != nil {
						return err
					}

				}

				if err := zf.Close(); err != nil {
					return err
				}

				if err := tf.Close(); err != nil {
					return err
				}
				path = tf.Name()
			}

			digest, err := createBlob(cmd, client, path)
			if err != nil {
				return err
			}

			modelfile = bytes.ReplaceAll(modelfile, []byte(c.Args), []byte("@"+digest))
		}
	}

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			spinner.Stop()

			bar, ok := bars[resp.Digest]
			if !ok {
				bar = progress.NewBar(fmt.Sprintf("pulling %s...", resp.Digest[7:19]), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			spinner.Stop()

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.CreateRequest{Name: args[0], Modelfile: string(modelfile)}
	if err := client.Create(cmd.Context(), &request, fn); err != nil {
		return err
	}

	return nil
}
