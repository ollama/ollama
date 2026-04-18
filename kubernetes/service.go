
// ListServices returns all services for models.
func (sm *ServiceManager) ListServices(ctx context.Context) ([]*corev1.Service, error) {
	services, err := sm.provider.clientset.CoreV1().Services(sm.provider.namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "app=ollama",
	})
	if err != nil {
		return nil, NewKubernetesError(
			ErrTypeDeploymentFailed,
			"failed to list services",
			err,
		)
	}

	result := make([]*corev1.Service, len(services.Items))
	for i := range services.Items {
		result[i] = &services.Items[i]
	}
	return result, nil
}

// generateServiceManifest creates a Kubernetes Service manifest.
func (sm *ServiceManager) generateServiceManifest(spec *ServiceSpec) *corev1.Service {
	svcType := corev1.ServiceTypeClusterIP
	if spec.Type != "" {
		svcType = spec.Type
	}

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      spec.Name,
			Namespace: sm.provider.namespace,
			Labels: map[string]string{
				"app":   "ollama",
				"model": spec.ModelName,
			},
		},
		Spec: corev1.ServiceSpec{
			Type:     svcType,
			Selector: spec.Selector,
			Ports: []corev1.ServicePort{
				{
					Name:       "api",
					Port:       spec.Port,
					TargetPort: intstr.FromInt(int(spec.Port)),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}

	return service
}

// createContext returns a context for service manager operations.
// This is a helper to ensure consistent context handling.
func (sm *ServiceManager) createContext() context.Context {
	return context.Background()
}
package kubernetes

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// ServiceManager handles Kubernetes service creation and management.
type ServiceManager struct {
	provider *Provider
}

// NewServiceManager creates a new service manager.
func NewServiceManager(provider *Provider) *ServiceManager {
	return &ServiceManager{
		provider: provider,
	}
}

// ServiceSpec holds service configuration.
type ServiceSpec struct {
	Name      string
	ModelName string
	Port      int32
	Selector  map[string]string
	Type      corev1.ServiceType
}

// CreateService creates a new Kubernetes service for a model.
func (sm *ServiceManager) CreateService(ctx context.Context, spec *ServiceSpec) (*corev1.Service, error) {
	// Validate inputs
	if spec == nil {
		return nil, NewKubernetesError(
			ErrTypeInvalidConfig,
			"service spec cannot be nil",
			fmt.Errorf("spec is required"),
		)
	}

	if spec.Name == "" {
		return nil, NewKubernetesError(
			ErrTypeInvalidConfig,
			"service name cannot be empty",
			fmt.Errorf("spec.Name is required"),
		)
	}

	if spec.Port <= 0 {
		return nil, NewKubernetesError(
			ErrTypeInvalidConfig,
			"service port must be greater than zero",
			fmt.Errorf("port=%d", spec.Port),
		).WithDetails("port", spec.Port)
	}

	// Build Service manifest
	service := sm.generateServiceManifest(spec)
	if service == nil {
		return nil, NewKubernetesError(
			ErrTypeInvalidConfig,
			"failed to generate service manifest",
			fmt.Errorf("generated manifest is nil"),
		)
	}

	// Create Service via API
	created, err := sm.provider.clientset.CoreV1().Services(sm.provider.namespace).Create(sm.createContext(), service, metav1.CreateOptions{})
	if err != nil {
		return nil, NewKubernetesError(
			ErrTypeDeploymentFailed,
			fmt.Sprintf("failed to create service %s", spec.Name),
			err,
		).WithDetails("service", spec.Name)
	}

	return created, nil
}

// UpdateService updates an existing service.
func (sm *ServiceManager) UpdateService(ctx context.Context, name string, service *corev1.Service) error {
	// TODO: Implement service update
	// 1. Get existing Service
	// 2. Apply updates
	// 3. Patch Service
	return nil
}

// DeleteService removes a service from Kubernetes.
func (sm *ServiceManager) DeleteService(ctx context.Context, name string) error {
	// Validate inputs
	if name == "" {
		return NewKubernetesError(
			ErrTypeInvalidConfig,
			"service name cannot be empty",
			fmt.Errorf("name is required"),
		)
	}

	// Delete Service by name
	err := sm.provider.clientset.CoreV1().Services(sm.provider.namespace).Delete(sm.createContext(), name, metav1.DeleteOptions{})
	if err != nil {
		return NewKubernetesError(
			ErrTypeDeploymentFailed,
			fmt.Sprintf("failed to delete service %s", name),
			err,
		).WithDetails("service", name)
	}

	return nil
}

// GetService retrieves a service by name.
func (sm *ServiceManager) GetService(ctx context.Context, name string) (*corev1.Service, error) {
	// TODO: Implement service retrieval
	service, err := sm.provider.clientset.CoreV1().Services(sm.provider.namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get service %s: %w", name, err)
	}
	return service, nil
}

// GetEndpoints retrieves the endpoints for a service.
func (sm *ServiceManager) GetEndpoints(ctx context.Context, name string) (*corev1.Endpoints, error) {
	// Validate inputs
	if name == "" {
		return nil, NewKubernetesError(
			ErrTypeInvalidConfig,
			"service name cannot be empty",
			fmt.Errorf("name is required"),
		)
	}

	// Get Endpoints by Service name
	endpoints, err := sm.provider.clientset.CoreV1().Endpoints(sm.provider.namespace).Get(sm.createContext(), name, metav1.GetOptions{})
	if err != nil {
		return nil, NewKubernetesError(
			ErrTypeNotFound,
			fmt.Sprintf("endpoints not found for service %s", name),
			err,
		).WithDetails("service", name)
	}

	return endpoints, nil
}
