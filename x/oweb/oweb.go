package oweb

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"

	"bllamo.com/client/ollama"
)

func Missing(field string) error {
	return &ollama.Error{
		Status:  400,
		Code:    "missing",
		Field:   field,
		Message: fmt.Sprintf("%s is required", field),
	}
}

func Invalid(field, value, format string, args ...any) error {
	return &ollama.Error{
		Status:  400,
		Code:    "invalid",
		Field:   field,
		Value:   value,
		Message: fmt.Sprintf(format, args...),
	}
}

// Convenience errors
var (
	ErrNotFound         = &ollama.Error{Status: 404, Code: "not_found"}
	ErrInternal         = &ollama.Error{Status: 500, Code: "internal_error"}
	ErrMethodNotAllowed = &ollama.Error{Status: 405, Code: "method_not_allowed"}
)

type HandlerFunc func(w http.ResponseWriter, r *http.Request) error

func Serve(h HandlerFunc, w http.ResponseWriter, r *http.Request) {
	if err := h(w, r); err != nil {
		// TODO: take a slog.Logger
		log.Printf("error: %v", err)
		var oe *ollama.Error
		if !errors.As(err, &oe) {
			oe = ErrInternal
		}
		oe.Status = cmp.Or(oe.Status, 400)
		w.WriteHeader(oe.Status)
		if err := EncodeJSON(w, oe); err != nil {
			log.Printf("error encoding error: %v", err)
		}
	}
}

func DecodeUserJSON[T any](field string, r io.Reader) (*T, error) {
	v, err := DecodeJSON[T](r)
	if err == nil {
		return v, nil
	}
	var msg string
	var e *json.SyntaxError
	if errors.As(err, &e) {
		msg = e.Error()
	}
	var se *json.UnmarshalTypeError
	if errors.As(err, &se) {
		msg = fmt.Sprintf("%s (%q) is not a %s", se.Field, se.Value, se.Type)
	}
	return nil, Invalid("invalid_json", field, "", msg)
}

func DecodeJSON[T any](r io.Reader) (*T, error) {
	var v *T
	if err := json.NewDecoder(r).Decode(&v); err != nil {
		var zero T
		return &zero, err
	}
	return v, nil
}

func EncodeJSON(w io.Writer, v any) error {
	return json.NewEncoder(w).Encode(v)
}
