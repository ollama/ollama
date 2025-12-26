package server

import (
	"encoding/json"
	"errors"

	"github.com/gin-gonic/gin"
)

func bindJSON(c *gin.Context, obj any) error {
	if c.Request.Body == nil {
		return errors.New("missing request body")
	}
	
	decoder := json.NewDecoder(c.Request.Body)
	decoder.DisallowUnknownFields()
	
	if err := decoder.Decode(obj); err != nil {
		return err
	}
	
	return nil
}
