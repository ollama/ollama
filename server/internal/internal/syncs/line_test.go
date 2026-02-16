//go:build goexperiment.synctest

package syncs

import (
	"bytes"
	"io"
	"math/rand/v2"
	"testing"
	"testing/synctest"
)

func TestPipelineReadWriterTo(t *testing.T) {
	for range 10 {
		synctest.Run(func() {
			q := NewRelayReader()

			tickets := []struct {
				io.WriteCloser
				s string
			}{
				{q.Take(), "you"},
				{q.Take(), " say hi,"},
				{q.Take(), " and "},
				{q.Take(), "I say "},
				{q.Take(), "hello"},
			}

			rand.Shuffle(len(tickets), func(i, j int) {
				tickets[i], tickets[j] = tickets[j], tickets[i]
			})

			var g Group
			for i, t := range tickets {
				g.Go(func() {
					defer t.Close()
					if i%2 == 0 {
						// Use [relayWriter.WriteString]
						io.WriteString(t.WriteCloser, t.s)
					} else {
						t.Write([]byte(t.s))
					}
				})
			}

			var got bytes.Buffer
			var copyErr error // checked at end
			g.Go(func() {
				_, copyErr = io.Copy(&got, q)
			})

			synctest.Wait()

			q.Close()
			g.Wait()

			if copyErr != nil {
				t.Fatal(copyErr)
			}

			want := "you say hi, and I say hello"
			if got.String() != want {
				t.Fatalf("got %q, want %q", got.String(), want)
			}
		})
	}
}
