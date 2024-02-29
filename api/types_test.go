package api

import (
	"encoding/json"
	"math"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKeepAliveParsingFromStruct(t *testing.T) {
	tests := []struct {
		name string
		req  *ChatRequest
		exp  *SessionDuration
	}{
		{
			name: "Positive Duration",
			req: &ChatRequest{
				KeepAlive: &SessionDuration{42 * time.Minute},
			},
			exp: &SessionDuration{42 * time.Minute},
		},
		{
			name: "Negative Duration",
			req: &ChatRequest{
				KeepAlive: &SessionDuration{-1 * time.Minute},
			},
			exp: &SessionDuration{math.MaxInt64},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ser, err := json.Marshal(test.req)
			require.NoError(t, err)

			var dec ChatRequest
			err = json.Unmarshal([]byte(ser), &dec)
			require.NoError(t, err)

			assert.Equal(t, test.exp, dec.KeepAlive)
		})
	}
}
func TestKeepAliveParsingFromJSON(t *testing.T) {
	tests := []struct {
		name string
		req  string
		exp  *SessionDuration
	}{
		{
			name: "Positive Integer",
			req:  `{ "keep_alive": 42 }`,
			exp:  &SessionDuration{42 * time.Second},
		},
		{
			name: "Positive Integer String",
			req:  `{ "keep_alive": "42m" }`,
			exp:  &SessionDuration{42 * time.Minute},
		},
		{
			name: "Negative Integer",
			req:  `{ "keep_alive": -1 }`,
			exp:  &SessionDuration{math.MaxInt64},
		},
		{
			name: "Negative Integer String",
			req:  `{ "keep_alive": "-1m" }`,
			exp:  &SessionDuration{math.MaxInt64},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var dec ChatRequest
			err := json.Unmarshal([]byte(test.req), &dec)
			require.NoError(t, err)

			assert.Equal(t, test.exp, dec.KeepAlive)
		})
	}
}

func TestKeepAliveParsingFromContext(t *testing.T) {
	tests := []struct {
		name string
		ctx  *gin.Context
		exp  *SessionDuration
	}{
		{
			name: "Positive Integer",
			ctx: func() *gin.Context {
				c := &gin.Context{}
				d, _ := NewSessionDuration(WithDuration(42 * time.Minute))
				c.Set("keepAlive", d)
				return c
			}(),
			exp: &SessionDuration{42 * time.Minute},
		},
		{
			name: "Negative Integer",
			ctx: func() *gin.Context {
				c := &gin.Context{}
				d, _ := NewSessionDuration(WithDuration(-1 * time.Minute))
				c.Set("keepAlive", d)
				return c
			}(),
			exp: &SessionDuration{math.MaxInt64},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			keepAliveStr, _ := test.ctx.Get("keepAlive")
			keepAlive := keepAliveStr.(*SessionDuration)

			assert.Equal(t, test.exp, keepAlive)
		})
	}
}
