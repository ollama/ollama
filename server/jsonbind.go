package server

import (
	"encoding/json"
	"errors"
	"io"

	"github.com/gin-gonic/gin"
)

// bindRequestJSON decodes a single JSON value from the request body and rejects
// trailing non-whitespace data after that value.
func bindRequestJSON(c *gin.Context, dst any) error {
	dec := json.NewDecoder(c.Request.Body)
	if err := dec.Decode(dst); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return errors.New("invalid character after top-level value")
		}
		return err
	}
	return nil
}
