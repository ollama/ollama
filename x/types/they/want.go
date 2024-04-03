package they

import (
	"net/http"
	"strings"
)

// Want returns true if the request method is method and the request path
// starts with pathPrefix.
func Want(r *http.Request, method string, pathPrefix string) bool {
	return r.Method == method && strings.HasPrefix(r.URL.Path, pathPrefix)
}
