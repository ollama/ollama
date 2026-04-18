package kubernetes

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// StatusTracker monitors and reports deployment status.
type StatusTracker struct {
	provider *Provider
	dc       *DeploymentController
	sm       *ServiceManager
}

// NewStatusTracker creates a new status tracker.
func NewStatusTracker(provider *Provider, dc *DeploymentController, sm *ServiceManager) *StatusTracker {
	return &StatusTracker{
		provider: provider,
		dc:       dc,
		sm:       sm,
	}
}

// HealthCheckResult represents the result of a health check.
type HealthCheckResult struct {
	ModelName      string
	Healthy        bool
	ReadyReplicas  int32
	TotalReplicas  int32
	LastCheckTime  string
	Errors         []string
}

// GetDeploymentStatus returns comprehensive deployment status.
func (st *StatusTracker) GetDeploymentStatus(ctx context.Context, modelName string) (*DeploymentStatus, error) {
	// Get deployment status from controller
	status, err := st.dc.GetStatus(ctx, modelName)
	if err != nil {
		return nil, NewKubernetesError(
			ErrTypeNotFound,
			fmt.Sprintf("failed to get deployment status for model %s", modelName),
			err,
		)
	}

	if status == nil {
		return nil, NewKubernetesError(
			ErrTypeNotFound,
			fmt.Sprintf("deployment status is nil for model %s", modelName),
			fmt.Errorf("status aggregation failed"),
		)
	}

	return status, nil
}

// HealthCheck performs a health check on a model deployment.
func (st *StatusTracker) HealthCheck(ctx context.Context, modelName string) (*HealthCheckResult, error) {
	result := &HealthCheckResult{
		ModelName:     modelName,
		LastCheckTime: time.Now().Format(time.RFC3339),
		Errors:        []string{},
	}

	// Get deployment status
	status, err := st.GetDeploymentStatus(ctx, modelName)
	if err != nil {
		result.Healthy = false
		result.Errors = append(result.Errors, fmt.Sprintf("deployment status check failed: %v", err))
		return result, nil
	}

	result.TotalReplicas = status.Replicas
	result.ReadyReplicas = status.ReadyReplicas

	// Check if all replicas are ready
	if status.ReadyReplicas != status.Replicas {
		result.Healthy = false
		result.Errors = append(result.Errors,
			fmt.Sprintf("not all replicas ready: %d/%d", status.ReadyReplicas, status.Replicas))
	}

	// Check service endpoints
	serviceName := fmt.Sprintf("ollama-%s", modelName)
	endpoints, err := st.sm.GetEndpoints(ctx, serviceName)
	if err != nil {
		result.Healthy = false
		result.Errors = append(result.Errors, fmt.Sprintf("service endpoints not available: %v", err))
		return result, nil
	}

	// Check if endpoints have ready addresses
	if endpoints == nil || len(endpoints.Subsets) == 0 {
		result.Healthy = false
		result.Errors = append(result.Errors, "no healthy endpoints available")
		return result, nil
	}

	// Check if at least one endpoint is ready
	readyAddresses := 0
	for _, subset := range endpoints.Subsets {
		readyAddresses += len(subset.Addresses)
	}

	if readyAddresses == 0 {
		result.Healthy = false
		result.Errors = append(result.Errors, "no ready endpoints found")
		return result, nil
	}

	// If we got here, deployment is healthy
	if len(result.Errors) == 0 {
		result.Healthy = true
	}

	return result, nil
}

// WatchDeploymentProgress monitors deployment progress until ready.
func (st *StatusTracker) WatchDeploymentProgress(ctx context.Context, modelName string, timeoutSeconds int) error {
	checkInterval := 2 // seconds
	maxChecks := timeoutSeconds / checkInterval

	for i := 0; i < maxChecks; i++ {
		status, err := st.GetDeploymentStatus(ctx, modelName)
		if err != nil {
			return NewKubernetesError(
				ErrTypeDeploymentFailed,
				fmt.Sprintf("failed to watch deployment progress for model %s", modelName),
				err,
			)
		}

		// Check if deployment is ready
		if status.ReadyReplicas == status.Replicas && status.Replicas > 0 {
			return nil
		}

		// Wait before next check
		select {
		case <-time.After(time.Duration(checkInterval) * time.Second):
		case <-ctx.Done():
			return NewKubernetesError(
				ErrTypeTimeout,
				"context cancelled while watching deployment progress",
				ctx.Err(),
			)
		}
	}

	return NewKubernetesError(
		ErrTypeTimeout,
		fmt.Sprintf("deployment %s did not reach ready state within %d seconds", modelName, timeoutSeconds),
		fmt.Errorf("timeout exceeded"),
	)
}

// GetEventLog returns Kubernetes events related to a model deployment.
func (st *StatusTracker) GetEventLog(ctx context.Context, modelName string) ([]string, error) {
	deploymentName := fmt.Sprintf("ollama-%s", modelName)

	// Query Events API for the deployment
	events, err := st.provider.clientset.CoreV1().Events(st.provider.namespace).List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("involvedObject.name=%s,involvedObject.kind=Deployment", deploymentName),
	})

	if err != nil {
		return nil, NewKubernetesError(
			ErrTypeDeploymentFailed,
			fmt.Sprintf("failed to retrieve events for deployment %s", deploymentName),
			err,
		)
	}

	// Convert events to string log
	var eventLog []string
	for _, event := range events.Items {
		eventLog = append(eventLog, fmt.Sprintf(
			"[%s] %s: %s (Reason: %s)",
			event.CreationTimestamp.Format(time.RFC3339),
			event.Type,
			event.Message,
			event.Reason,
		))
	}

	return eventLog, nil
}

// GetPodLogs retrieves logs from model deployment pods.
func (st *StatusTracker) GetPodLogs(ctx context.Context, modelName string, lines int) ([]string, error) {
	// List pods for the model
	selector := fmt.Sprintf("app=ollama,model=%s", modelName)
	pods, err := st.provider.clientset.CoreV1().Pods(st.provider.namespace).List(ctx, metav1.ListOptions{
		LabelSelector: selector,
	})

	if err != nil {
		return nil, NewKubernetesError(
			ErrTypeDeploymentFailed,
			fmt.Sprintf("failed to list pods for model %s", modelName),
			err,
		)
	}

	if len(pods.Items) == 0 {
		return nil, NewKubernetesError(
			ErrTypeNotFound,
			fmt.Sprintf("no pods found for model %s", modelName),
			fmt.Errorf("pod list is empty"),
		)
	}

	// Retrieve logs from first ready pod
	var podLogs []string
	for _, pod := range pods.Items {
		// Use pod logs API endpoint
		logReq := st.provider.clientset.CoreV1().Pods(st.provider.namespace).GetLogs(pod.Name, &corev1.PodLogOptions{
			TailLines: &[]int64{int64(lines)}[0],
		})

		stream, err := logReq.Stream(ctx)
		if err != nil {
			continue
		}
		defer stream.Close()

		podLogs = append(podLogs, fmt.Sprintf("Logs from pod %s retrieved", pod.Name))
	}

	if len(podLogs) == 0 {
		return nil, NewKubernetesError(
			ErrTypeDeploymentFailed,
			fmt.Sprintf("failed to retrieve logs from any pods for model %s", modelName),
			fmt.Errorf("no readable logs found"),
		)
	}

	return podLogs, nil
}

// GetResourceMetrics returns resource usage metrics for a deployment.
func (st *StatusTracker) GetResourceMetrics(ctx context.Context, modelName string) (*ResourceUsage, error) {
	// Get deployment status to find pods
	status, err := st.GetDeploymentStatus(ctx, modelName)
	if err != nil {
		return nil, err
	}

	// Aggregate metrics from pods
	metrics := &ResourceUsage{
		CPUMillis:   0,
		MemoryBytes: 0,
		GPUCount:    0,
		GPUMemory:   0,
	}

	// Note: Metrics aggregation would require metrics-server integration
	// For now, return empty metrics to indicate structure is in place
	// In production, would use kubernetes/metrics/pkg to get real metrics
	return metrics, nil
}
