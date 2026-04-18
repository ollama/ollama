// waitForDeployment waits for a deployment to reach desired state.
func (dc *DeploymentController) waitForDeployment(ctx context.Context, modelName string, timeoutSeconds int) error {
	deploymentName := fmt.Sprintf("ollama-%s", modelName)

	// Simple polling mechanism with timeout
	checkInterval := 2 // seconds
	maxChecks := timeoutSeconds / checkInterval

	for i := 0; i < maxChecks; i++ {
		deployment, err := dc.provider.clientset.AppsV1().Deployments(dc.provider.namespace).Get(ctx, deploymentName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get deployment status: %w", err)
		}

		// Check if all replicas are ready
		if deployment.Status.ReadyReplicas == *deployment.Spec.Replicas {
			return nil
		}

		// Check if deployment has failed
		for _, cond := range deployment.Status.Conditions {
			if cond.Type == "Progressing" && cond.Status == "False" {
				return NewKubernetesError(
					ErrTypeDeploymentFailed,
					fmt.Sprintf("deployment %s failed to progress", deploymentName),
					fmt.Errorf("reason: %s", cond.Reason),
				)
			}
		}

		// Wait before next check
		select {
		case <-time.After(time.Duration(checkInterval) * time.Second):
		case <-ctx.Done():
			return NewKubernetesError(
				ErrTypeTimeout,
				"context cancelled while waiting for deployment",
				ctx.Err(),
			)
		}
	}

	return NewKubernetesError(
		ErrTypeTimeout,
		fmt.Sprintf("deployment %s did not reach desired state within %d seconds", deploymentName, timeoutSeconds),
		fmt.Errorf("timeout exceeded"),
	)
}

// parseQuantity parses a Kubernetes resource quantity string.
// Examples: "2", "100m", "8Gi", "1024Mi"
func parseQuantity(value string) resource.Quantity {
	q, _ := resource.ParseQuantity(value)
	return q
}
package kubernetes

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// DeploymentController manages model deployments in Kubernetes.
type DeploymentController struct {
	provider *Provider
}

// DeploymentStatus represents the status of a model deployment.
type DeploymentStatus struct {
	ModelName      string
	State          string
	Replicas       int32
	ReadyReplicas  int32
	ServiceName    string
	CreatedAt      metav1.Time
	LastUpdated    metav1.Time
	Conditions     []string
	ResourceUsage  *ResourceUsage
}

// ResourceUsage tracks current resource usage.
type ResourceUsage struct {
	CPUMillis    int64
	MemoryBytes  int64
	GPUCount     int
	GPUMemory    int64
}

// NewDeploymentController creates a new deployment controller.
func NewDeploymentController(provider *Provider) *DeploymentController {
	return &DeploymentController{
		provider: provider,
	}
}

// Deploy creates a new model deployment in Kubernetes.
func (dc *DeploymentController) Deploy(ctx context.Context, modelName, version string, replicas int32) error {
	// Validate inputs
	if modelName == "" {
		return NewKubernetesError(
			ErrTypeInvalidConfig,
			"model name cannot be empty",
			fmt.Errorf("modelName is required"),
		)
	}

	if replicas <= 0 {
		return NewKubernetesError(
			ErrTypeInvalidConfig,
			"replicas must be greater than zero",
			fmt.Errorf("replicas=%d", replicas),
		).WithDetails("replicas", replicas)
	}

	// Create PersistentVolumeClaim for model storage
	sm := NewStorageManager(dc.provider)
	pvcSpec := &PVCSpec{
		Name:      fmt.Sprintf("ollama-%s-pvc", modelName),
		ModelName: modelName,
		Size:      "50Gi",
	}
	pvc, err := sm.CreatePVC(ctx, pvcSpec)
	if err != nil {
		return NewKubernetesError(
			ErrTypeStorageError,
			fmt.Sprintf("failed to create PVC for model %s", modelName),
			err,
		)
	}

	// Wait for PVC to be bound
	if err := sm.WaitForPVCBound(ctx, pvcSpec.Name, 300); err != nil {
		return NewKubernetesError(
			ErrTypeStorageError,
			fmt.Sprintf("timeout waiting for PVC %s to bind", pvcSpec.Name),
			err,
		)
	}

	// Generate Deployment manifest
	deployment := dc.generateDeploymentManifest(modelName, version, replicas)
	if deployment == nil {
		return NewKubernetesError(
			ErrTypeDeploymentFailed,
			fmt.Sprintf("failed to generate deployment manifest for model %s", modelName),
			fmt.Errorf("generated manifest is nil"),
		)
	}

	// Create Service for model access
	svcMgr := NewServiceManager(dc.provider)
	svcSpec := &ServiceSpec{
		Name:      fmt.Sprintf("ollama-%s", modelName),
		ModelName: modelName,
		Port:      11434,
		Selector: map[string]string{
			"app":   "ollama",
			"model": modelName,
		},
	}
	_, err = svcMgr.CreateService(ctx, svcSpec)
	if err != nil {
		return NewKubernetesError(
			ErrTypeDeploymentFailed,
			fmt.Sprintf("failed to create service for model %s", modelName),
			err,
		)
	}

	// Create Deployment
	deploymentResult, err := dc.provider.clientset.AppsV1().Deployments(dc.provider.namespace).Create(ctx, deployment, metav1.CreateOptions{})
	if err != nil {
		return NewKubernetesError(
			ErrTypeDeploymentFailed,
			fmt.Sprintf("failed to create deployment for model %s", modelName),
			err,
		).WithDetails("deployment", deployment.Name)
	}

	// Wait for deployment to be ready
	if err := dc.waitForDeployment(ctx, modelName, 600); err != nil {
		return NewKubernetesError(
			ErrTypeTimeout,
			fmt.Sprintf("deployment %s did not become ready in time", deploymentResult.Name),
			err,
		)
	}

	return nil
}

// Undeploy removes a model deployment from Kubernetes.
func (dc *DeploymentController) Undeploy(ctx context.Context, modelName string) error {
	// Validate inputs
	if modelName == "" {
		return NewKubernetesError(
			ErrTypeInvalidConfig,
			"model name cannot be empty",
			fmt.Errorf("modelName is required"),
		)
	}

	// Delete Service
	svcMgr := NewServiceManager(dc.provider)
	if err := svcMgr.DeleteService(ctx, fmt.Sprintf("ollama-%s", modelName)); err != nil {
		// Log error but continue with deployment removal
		_ = err
	}

	// Delete Deployment
	deploymentName := fmt.Sprintf("ollama-%s", modelName)
	deletePolicy := metav1.DeletePropagationForeground
	if err := dc.provider.clientset.AppsV1().Deployments(dc.provider.namespace).Delete(ctx, deploymentName, metav1.DeleteOptions{
		PropagationPolicy: &deletePolicy,
	}); err != nil {
		return NewKubernetesError(
			ErrTypeDeploymentFailed,
			fmt.Sprintf("failed to delete deployment for model %s", modelName),
			err,
		).WithDetails("deployment", deploymentName)
	}

	// Delete PVC
	sm := NewStorageManager(dc.provider)
	pvcName := fmt.Sprintf("ollama-%s-pvc", modelName)
	if err := sm.DeletePVC(ctx, pvcName); err != nil {
		return NewKubernetesError(
			ErrTypeStorageError,
			fmt.Sprintf("failed to delete PVC for model %s", modelName),
			err,
		).WithDetails("pvc", pvcName)
	}

	return nil
}

// GetStatus returns the current status of a model deployment.
func (dc *DeploymentController) GetStatus(ctx context.Context, modelName string) (*DeploymentStatus, error) {
	// Get Deployment
	deploymentName := fmt.Sprintf("ollama-%s", modelName)
	deployment, err := dc.provider.clientset.AppsV1().Deployments(dc.provider.namespace).Get(ctx, deploymentName, metav1.GetOptions{})
	if err != nil {
		return nil, NewKubernetesError(
			ErrTypeNotFound,
			fmt.Sprintf("deployment not found for model %s", modelName),
			err,
		).WithDetails("deployment", deploymentName)
	}

	// Get Service endpoints
	svcMgr := NewServiceManager(dc.provider)
	serviceName := fmt.Sprintf("ollama-%s", modelName)
	service, err := svcMgr.GetService(ctx, serviceName)
	if err != nil {
		return nil, NewKubernetesError(
			ErrTypeNotFound,
			fmt.Sprintf("service not found for model %s", modelName),
			err,
		)
	}

	// Build status
	status := &DeploymentStatus{
		ModelName:     modelName,
		State:         "running",
		Replicas:      *deployment.Spec.Replicas,
		ReadyReplicas: deployment.Status.ReadyReplicas,
		ServiceName:   service.Name,
		CreatedAt:     deployment.ObjectMeta.CreationTimestamp,
		LastUpdated:   deployment.ObjectMeta.ManagedFields[0].Time,
	}

	// Add conditions as strings
	for _, cond := range deployment.Status.Conditions {
		status.Conditions = append(status.Conditions, fmt.Sprintf("%s: %s", cond.Type, cond.Status))
	}

	return status, nil
}

// Scale changes the number of replicas for a model deployment.
func (dc *DeploymentController) Scale(ctx context.Context, modelName string, replicas int32) error {
	// Validate inputs
	if modelName == "" {
		return NewKubernetesError(
			ErrTypeInvalidConfig,
			"model name cannot be empty",
			fmt.Errorf("modelName is required"),
		)
	}

	if replicas <= 0 {
		return NewKubernetesError(
			ErrTypeInvalidConfig,
			"replicas must be greater than zero",
			fmt.Errorf("replicas=%d", replicas),
		).WithDetails("replicas", replicas)
	}

	// Get current Deployment
	deploymentName := fmt.Sprintf("ollama-%s", modelName)
	deployment, err := dc.provider.clientset.AppsV1().Deployments(dc.provider.namespace).Get(ctx, deploymentName, metav1.GetOptions{})
	if err != nil {
		return NewKubernetesError(
			ErrTypeNotFound,
			fmt.Sprintf("deployment not found for model %s", modelName),
			err,
		).WithDetails("deployment", deploymentName)
	}

	// Update replica count
	deployment.Spec.Replicas = &replicas

	// Apply update
	_, err = dc.provider.clientset.AppsV1().Deployments(dc.provider.namespace).Update(ctx, deployment, metav1.UpdateOptions{})
	if err != nil {
		return NewKubernetesError(
			ErrTypeDeploymentFailed,
			fmt.Sprintf("failed to scale deployment for model %s to %d replicas", modelName, replicas),
			err,
		).WithDetails("replicas", replicas)
	}

	// Monitor rollout
	if err := dc.waitForDeployment(ctx, modelName, 300); err != nil {
		return NewKubernetesError(
			ErrTypeTimeout,
			fmt.Sprintf("deployment %s did not complete scaling in time", deploymentName),
			err,
		)
	}

	return nil
}

// generateDeploymentManifest creates a Kubernetes Deployment manifest for a model.
func (dc *DeploymentController) generateDeploymentManifest(modelName, version string, replicas int32) *appsv1.Deployment {
	labels := map[string]string{
		"app":     "ollama",
		"model":   modelName,
		"version": version,
	}

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("ollama-%s", modelName),
			Namespace: dc.provider.namespace,
			Labels:    labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "ollama",
							Image: fmt.Sprintf("ollama:latest"),
							Ports: []corev1.ContainerPort{
								{
									Name:          "api",
									ContainerPort: 11434,
									Protocol:      corev1.ProtocolTCP,
								},
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    parseQuantity("2"),
									corev1.ResourceMemory: parseQuantity("8Gi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    parseQuantity("4"),
									corev1.ResourceMemory: parseQuantity("16Gi"),
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "model-storage",
									MountPath: "/models",
								},
							},
							LivenessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path:   "/api/health",
										Port:   intstr.FromInt(11434),
										Scheme: corev1.URISchemeHTTP,
									},
								},
								InitialDelaySeconds: 30,
								PeriodSeconds:       10,
								TimeoutSeconds:      5,
								FailureThreshold:    3,
							},
							ReadinessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path:   "/api/health",
										Port:   intstr.FromInt(11434),
										Scheme: corev1.URISchemeHTTP,
									},
								},
								InitialDelaySeconds: 15,
								PeriodSeconds:       5,
								TimeoutSeconds:      3,
								FailureThreshold:    2,
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "model-storage",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: fmt.Sprintf("ollama-%s-pvc", modelName),
								},
							},
						},
					},
					RestartPolicy: corev1.RestartPolicyAlways,
				},
			},
		},
	}

	return deployment
