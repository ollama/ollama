package kubernetes

import (
	"context"
	"fmt"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

// Provider manages Kubernetes operations for Ollama models.
type Provider struct {
	clientset    kubernetes.Interface
	config       *rest.Config
	namespace    string
	storageClass string
}

// NewProvider creates a new Kubernetes provider.
func NewProvider(kubeconfig, namespace, storageClass string) (*Provider, error) {
	config, err := getKubeConfig(kubeconfig)
	if err != nil {
		return nil, fmt.Errorf("failed to load kubeconfig: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	return &Provider{
		clientset:    clientset,
		config:       config,
		namespace:    namespace,
		storageClass: storageClass,
	}, nil
}

// Connect verifies connectivity to the Kubernetes cluster.
func (p *Provider) Connect(ctx context.Context) error {
	if p.clientset == nil {
		return NewKubernetesError(
			ErrTypeAuthFailed,
			"kubernetes client not initialized",
			fmt.Errorf("clientset is nil"),
		)
	}

	// Verify connectivity by querying server version
	_, err := p.clientset.Discovery().ServerVersion()
	if err != nil {
		return NewKubernetesError(
			ErrTypeClusterUnavailable,
			"failed to connect to kubernetes cluster",
			err,
		).WithDetails("endpoint", p.config.Host)
	}

	return nil
}

// Disconnect performs cleanup operations.
func (p *Provider) Disconnect() error {
	// No explicit cleanup needed for client-go
	// Resources are cleaned up by Go's garbage collector
	return nil
}

// IsAvailable returns true if the Kubernetes cluster is available.
func (p *Provider) IsAvailable(ctx context.Context) bool {
	if p.clientset == nil {
		return false
	}

	// Quick health check using discovery client
	_, err := p.clientset.Discovery().ServerVersion()
	return err == nil
}

// getKubeConfig loads the kubeconfig file.
func getKubeConfig(kubeconfig string) (*rest.Config, error) {
	if kubeconfig == "" {
		// Try in-cluster config first
		config, err := rest.InClusterConfig()
		if err == nil {
			return config, nil
		}
		// Fall back to default kubeconfig location
		kubeconfig = clientcmd.RecommendedHomeFile
	}

	return clientcmd.BuildConfigFromFlags("", kubeconfig)
}
