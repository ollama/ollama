package server

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/models"
	"github.com/ollama/ollama/monitoring"
)

var (
	modelManager    *models.ModelManager
	metricsCollector *monitoring.MetricsCollector
)

func init() {
	modelManager = models.NewModelManager("/models")
	metricsCollector = monitoring.NewMetricsCollector()
}

// Model Management Handlers (Phase 8)

func (s *Server) ListAvailableModelsHandler(c *gin.Context) {
	availableModels := modelManager.ListAvailableModels()
	c.JSON(http.StatusOK, gin.H{
		"models": availableModels,
		"count":  len(availableModels),
	})
}

func (s *Server) ListLocalModelsHandler(c *gin.Context) {
	localModels, err := modelManager.ListLocalModels()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"models": localModels,
		"count":  len(localModels),
	})
}

func (s *Server) GetModelInfoHandler(c *gin.Context) {
	modelID := c.Param("id")

	modelInfo, err := modelManager.GetModelInfo(modelID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, modelInfo)
}

func (s *Server) DownloadModelHandler(c *gin.Context) {
	var req struct {
		ModelID     string `json:"model_id" binding:"required"`
		DownloadURL string `json:"download_url" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx := c.Request.Context()
	if err := modelManager.DownloadModel(ctx, req.ModelID, req.DownloadURL); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":  "download started",
		"model_id": req.ModelID,
	})
}

func (s *Server) GetDownloadProgressHandler(c *gin.Context) {
	modelID := c.Param("id")

	progress, err := modelManager.GetDownloadProgress(modelID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	// Calculate percentage
	percentage := float64(0)
	if progress.TotalBytes > 0 {
		percentage = float64(progress.BytesDownloaded) / float64(progress.TotalBytes) * 100
	}

	c.JSON(http.StatusOK, gin.H{
		"model_id":         progress.ModelID,
		"status":           progress.Status,
		"bytes_downloaded": progress.BytesDownloaded,
		"total_bytes":      progress.TotalBytes,
		"percentage":       percentage,
		"speed":            progress.Speed,
		"eta":              progress.ETA,
		"started_at":       progress.StartedAt,
		"error":            progress.Error,
	})
}

func (s *Server) CancelDownloadHandler(c *gin.Context) {
	modelID := c.Param("id")

	if err := modelManager.CancelDownload(modelID); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":  "download cancelled",
		"model_id": modelID,
	})
}

func (s *Server) DeleteModelHandler(c *gin.Context) {
	modelID := c.Param("id")

	if err := modelManager.DeleteModel(modelID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":  "model deleted",
		"model_id": modelID,
	})
}

func (s *Server) CompareModelsHandler(c *gin.Context) {
	modelID1 := c.Query("model1")
	modelID2 := c.Query("model2")

	if modelID1 == "" || modelID2 == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "both model1 and model2 parameters are required"})
		return
	}

	comparison, err := modelManager.CompareModels(modelID1, modelID2)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, comparison)
}

func (s *Server) GetStorageStatsHandler(c *gin.Context) {
	stats, err := modelManager.GetStorageStats()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, stats)
}

// Performance Monitoring Handlers (Phase 9)

func (s *Server) GetPerformanceMetricsHandler(c *gin.Context) {
	metrics := metricsCollector.GetPerformanceMetrics()
	c.JSON(http.StatusOK, metrics)
}

func (s *Server) GetProviderMetricsHandler(c *gin.Context) {
	provider := c.Param("provider")

	metrics := metricsCollector.GetProviderMetrics(provider)
	if metrics == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "provider not found"})
		return
	}

	c.JSON(http.StatusOK, metrics)
}

func (s *Server) GetSystemMetricsHandler(c *gin.Context) {
	metrics := metricsCollector.GetSystemMetrics()
	c.JSON(http.StatusOK, metrics)
}

func (s *Server) GetTopProvidersHandler(c *gin.Context) {
	limit := 5
	if limitStr := c.Query("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}

	topProviders := metricsCollector.GetTopProviders(limit)
	c.JSON(http.StatusOK, gin.H{
		"providers": topProviders,
		"count":     len(topProviders),
	})
}

func (s *Server) GetCostBreakdownHandler(c *gin.Context) {
	breakdown := metricsCollector.GetCostBreakdown()
	c.JSON(http.StatusOK, gin.H{
		"breakdown": breakdown,
	})
}

func (s *Server) GetAlertsHandler(c *gin.Context) {
	alerts := metricsCollector.GetAlerts()
	c.JSON(http.StatusOK, gin.H{
		"alerts": alerts,
		"count":  len(alerts),
	})
}

func (s *Server) ResetMetricsHandler(c *gin.Context) {
	metricsCollector.Reset()
	c.JSON(http.StatusOK, gin.H{
		"message": "metrics reset successfully",
	})
}

func (s *Server) RecordCustomMetricHandler(c *gin.Context) {
	var metric monitoring.Metric
	if err := c.ShouldBindJSON(&metric); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	metricsCollector.RecordMetric(metric)
	c.JSON(http.StatusOK, gin.H{
		"message": "metric recorded",
	})
}
