
// generatePVCManifest creates a PVC manifest.
func (sm *StorageManager) generatePVCManifest(spec *PVCSpec) *corev1.PersistentVolumeClaim {
	accessMode := spec.AccessMode
	if accessMode == "" {
		accessMode = corev1.ReadWriteOnce
	}

	quantity := parseQuantity(spec.Size)

	pvc := &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      spec.Name,
			Namespace: sm.provider.namespace,
			Labels: map[string]string{
				"app":   "ollama",
				"model": spec.ModelName,
			},
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes:      []corev1.PersistentVolumeAccessMode{accessMode},
			StorageClassName: &sm.provider.storageClass,
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: quantity,
				},
			},
		},
	}

	return pvc
}

// parseQuantity parses a Kubernetes resource quantity string.
func parseQuantity(value string) resource.Quantity {
	q, _ := resource.ParseQuantity(value)
	return q
}
package kubernetes

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// StorageManager handles persistent storage for model deployments.
type StorageManager struct {
	provider *Provider
}

// NewStorageManager creates a new storage manager.
func NewStorageManager(provider *Provider) *StorageManager {
	return &StorageManager{
		provider: provider,
	}
}

// PVCSpec holds PersistentVolumeClaim specifications.
type PVCSpec struct {
	Name      string
	ModelName string
	Size      string // e.g., "50Gi"
	AccessMode corev1.PersistentVolumeAccessMode
}

// CreatePVC creates a PersistentVolumeClaim for model storage.
func (sm *StorageManager) CreatePVC(ctx context.Context, spec *PVCSpec) (*corev1.PersistentVolumeClaim, error) {
	// Validate inputs
	if spec == nil {
		return nil, NewKubernetesError(
			ErrTypeInvalidConfig,
			"PVC spec cannot be nil",
			fmt.Errorf("spec is required"),
		)
	}

	if spec.Name == "" {
		return nil, NewKubernetesError(
			ErrTypeInvalidConfig,
			"PVC name cannot be empty",
			fmt.Errorf("spec.Name is required"),
		)
	}

	if spec.Size == "" {
		return nil, NewKubernetesError(
			ErrTypeInvalidConfig,
			"PVC size cannot be empty",
			fmt.Errorf("spec.Size is required"),
		)
	}

	// Build PVC manifest
	pvc := sm.generatePVCManifest(spec)
	if pvc == nil {
		return nil, NewKubernetesError(
			ErrTypeInvalidConfig,
			"failed to generate PVC manifest",
			fmt.Errorf("generated manifest is nil"),
		)
	}

	// Create PVC via API
	created, err := sm.provider.clientset.CoreV1().PersistentVolumeClaims(sm.provider.namespace).Create(ctx, pvc, metav1.CreateOptions{})
	if err != nil {
		return nil, NewKubernetesError(
			ErrTypeStorageError,
			fmt.Sprintf("failed to create PVC %s", spec.Name),
			err,
		).WithDetails("pvc", spec.Name)
	}

	return created, nil
}

// DeletePVC removes a PersistentVolumeClaim.
func (sm *StorageManager) DeletePVC(ctx context.Context, name string) error {
	// Validate inputs
	if name == "" {
		return NewKubernetesError(
			ErrTypeInvalidConfig,
			"PVC name cannot be empty",
			fmt.Errorf("name is required"),
		)
	}

	// Delete PVC by name
	err := sm.provider.clientset.CoreV1().PersistentVolumeClaims(sm.provider.namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil {
		return NewKubernetesError(
			ErrTypeStorageError,
			fmt.Sprintf("failed to delete PVC %s", name),
			err,
		).WithDetails("pvc", name)
	}

	return nil
}

// GetPVC retrieves a PVC by name.
func (sm *StorageManager) GetPVC(ctx context.Context, name string) (*corev1.PersistentVolumeClaim, error) {
	// TODO: Implement PVC retrieval
	pvc, err := sm.provider.clientset.CoreV1().PersistentVolumeClaims(sm.provider.namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get PVC %s: %w", name, err)
	}
	return pvc, nil
}

// WaitForPVCBound waits for a PVC to be bound to a PV.
// timeoutSeconds parameter controls how long to wait before giving up.
func (sm *StorageManager) WaitForPVCBound(ctx context.Context, name string, timeoutSeconds int) error {
	checkInterval := 2 // seconds
	maxChecks := timeoutSeconds / checkInterval

	for i := 0; i < maxChecks; i++ {
		pvc, err := sm.provider.clientset.CoreV1().PersistentVolumeClaims(sm.provider.namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return NewKubernetesError(
				ErrTypeStorageError,
				fmt.Sprintf("failed to get PVC status for %s", name),
				err,
			)
		}

		if pvc.Status.Phase == corev1.ClaimBound {
			return nil
		}
}
		// Wait before next check
		select {
		case <-time.After(time.Duration(checkInterval) * time.Second):
		case <-ctx.Done():
			return NewKubernetesError(
				ErrTypeTimeout,
				"context cancelled while waiting for PVC binding",
				ctx.Err(),
			)
		}
	}

	return NewKubernetesError(
		ErrTypeTimeout,
		fmt.Sprintf("PVC %s did not bind within %d seconds", name, timeoutSeconds),
		fmt.Errorf("timeout exceeded"),
	)

// GetStorageUsage returns the current storage usage for a PVC.
func (sm *StorageManager) GetStorageUsage(ctx context.Context, name string) (*resource.Quantity, error) {
	// TODO: Implement usage retrieval
	// 1. Query PVC metrics
	// 2. Return current usage
	return nil, fmt.Errorf("not implemented")
}

// ListPVCs returns all model storage PVCs.
func (sm *StorageManager) ListPVCs(ctx context.Context) ([]*corev1.PersistentVolumeClaim, error) {
	// TODO: Implement PVC listing
	// 1. List all PVCs with model labels
	// 2. Filter by app=ollama
	// 3. Return list
	return nil, fmt.Errorf("not implemented")
}
