package server

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	internalcloud "github.com/ollama/ollama/internal/cloud"
)

type cloudShowResponse struct {
	StatusCode int
	Headers    http.Header
	Body       []byte
}

func newJSONCloudShowResponse(statusCode int, payload any) *cloudShowResponse {
	body, err := json.Marshal(payload)
	if err != nil {
		body = []byte(`{"error":"internal server error"}`)
	}

	headers := make(http.Header)
	headers.Set("Content-Type", "application/json; charset=utf-8")

	return &cloudShowResponse{
		StatusCode: statusCode,
		Headers:    headers,
		Body:       body,
	}
}

func cloudUnauthorizedShowResponse() *cloudShowResponse {
	payload := gin.H{"error": "unauthorized"}
	if signinURL, err := cloudProxySigninURL(); err == nil {
		payload["signin_url"] = signinURL
	}
	return newJSONCloudShowResponse(http.StatusUnauthorized, payload)
}

func writeCloudShowResponse(c *gin.Context, resp *cloudShowResponse) {
	if resp == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "internal server error"})
		return
	}

	copyProxyResponseHeaders(c.Writer.Header(), resp.Headers)
	c.Status(resp.StatusCode)
	if len(resp.Body) > 0 {
		_, _ = c.Writer.Write(resp.Body)
	}
}

func (s *Server) resolveCloudShow(c *gin.Context, req api.ShowRequest) (*api.ShowResponse, *cloudShowResponse) {
	if disabled, _ := internalcloud.Status(); disabled {
		return nil, newJSONCloudShowResponse(http.StatusForbidden, gin.H{
			"error": internalcloud.DisabledError(cloudErrRemoteModelDetailsUnavailable),
		})
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, newJSONCloudShowResponse(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}

	outReq, err := buildCloudProxyRequest(c, c.Request.URL.Path, body)
	if err != nil {
		return nil, newJSONCloudShowResponse(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}

	if err := cloudProxySignRequest(outReq.Context(), outReq); err != nil {
		slog.Warn("cloud proxy signing failed", "error", err)
		return nil, cloudUnauthorizedShowResponse()
	}

	resp, err := http.DefaultClient.Do(outReq)
	if err != nil {
		return nil, newJSONCloudShowResponse(http.StatusBadGateway, gin.H{"error": err.Error()})
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, newJSONCloudShowResponse(http.StatusBadGateway, gin.H{"error": err.Error()})
	}

	if resp.StatusCode >= http.StatusBadRequest {
		return nil, &cloudShowResponse{
			StatusCode: resp.StatusCode,
			Headers:    resp.Header.Clone(),
			Body:       respBody,
		}
	}

	var showResp api.ShowResponse
	if len(respBody) > 0 {
		if err := json.Unmarshal(respBody, &showResp); err != nil {
			return nil, newJSONCloudShowResponse(http.StatusBadGateway, gin.H{"error": err.Error()})
		}
	}

	return &showResp, nil
}
