package oweb

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

type Error struct {
	Status  int    `json:"-"`
	Code    string `json:"code"`
	Message string `json:"message"`
	Field   string `json:"field,omitempty"`
	RawBody []byte `json:"-"`
}

func Missing(field string) error {
	return &Error{
		Status:  400,
		Code:    "missing",
		Field:   field,
		Message: fmt.Sprintf("%s is required", field),
	}
}

func Mistake(code, field, message string) error {
	return &Error{
		Status:  400,
		Code:    code,
		Field:   field,
		Message: fmt.Sprintf("%s: %s", field, message),
	}
}

func Fault(code, message string) error {
	return &Error{
		Status:  500,
		Code:    "fault",
		Message: message,
	}
}

func (e *Error) Error() string {
	var b strings.Builder
	b.WriteString("ollama: ")
	b.WriteString(e.Code)
	if e.Message != "" {
		b.WriteString(": ")
		b.WriteString(e.Message)
	}
	return b.String()
}

// Convinience errors
var (
	ErrNotFound         = &Error{Status: 404, Code: "not_found"}
	ErrInternal         = &Error{Status: 500, Code: "internal_error"}
	ErrMethodNotAllowed = &Error{Status: 405, Code: "method_not_allowed"}
)

type HandlerFunc func(w http.ResponseWriter, r *http.Request) error

func Serve(h HandlerFunc, w http.ResponseWriter, r *http.Request) {
	if err := h(w, r); err != nil {
		// TODO: take a slog.Logger
		log.Printf("error: %v", err)
		var e *Error
		if !errors.As(err, &e) {
			e = ErrInternal
		}
		w.WriteHeader(cmp.Or(e.Status, 400))
		if err := EncodeJSON(w, e); err != nil {
			log.Printf("error encoding error: %v", err)
		}
	}
}

func DecodeUserJSON[T any](r io.Reader) (*T, error) {
	v, err := DecodeJSON[T](r)
	var e *json.SyntaxError
	if errors.As(err, &e) {
		return nil, &Error{Code: "invalid_json", Message: e.Error()}
	}
	var se *json.UnmarshalTypeError
	if errors.As(err, &se) {
		return nil, &Error{
			Code:    "invalid_json",
			Message: fmt.Sprintf("%s (%q) is not a %s", se.Field, se.Value, se.Type),
		}
	}
	return v, err
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

func Do[Res any](ctx context.Context, method, urlStr string, in any) (*Res, error) {
	var body bytes.Buffer
	if err := EncodeJSON(&body, in); err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, method, urlStr, &body)
	if err != nil {
		return nil, err
	}

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode/100 != 2 {
		var b bytes.Buffer
		body := io.TeeReader(res.Body, &b)
		e, err := DecodeJSON[Error](body)
		if err != nil {
			return nil, err
		}
		e.RawBody = b.Bytes()
		return nil, e
	}

	return DecodeJSON[Res](res.Body)
}
