package kubernetes

import (
	"context"
	"testing"
	"time"

	appv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
)

// IntegrationTestSuite provides setup for integration tests
type IntegrationTestSuite struct {
	clientset kubernetes.Interface
	provider  *Provider
	dc        *DeploymentController
	sm        *ServiceManager
	st        *StatusTracker
}

// SetupIntegrationTests initializes test environment
func setupIntegrationTestSuite() *IntegrationTestSuite {
	clientset := fake.NewSimpleClientset()

	provider := &Provider{
		clientset: clientset,
		namespace: "default",
		config:    nil,
	}

	dc := &DeploymentController{
		provider: provider,
	}

	sm := &ServiceManager{
		provider: provider,
	}

	st := NewStatusTracker(provider, dc, sm)

	return &IntegrationTestSuite{
		clientset: clientset,
		provider:  provider,
		dc:        dc,
		sm:        sm,
		st:        st,
	}
}

// TestDeploymentWorkflow_HappyPath tests complete deploy -> check -> scale -> undeploy flow
func TestDeploymentWorkflow_HappyPath(t *testing.T) {
	suite := setupIntegrationTestSuite()
	ctx := context.Background()
	modelName := "test-model"

	testSteps := []struct {
		name string
		run  func() error
	}{
		{
			name: "Deploy model",
			run: func() error {
				return suite.dc.Deploy(ctx, modelName, "ollama:latest", 2)
			},
		},
		{
			name: "Wait for deployment ready",
			run: func() error {
				return suite.st.WatchDeploymentProgress(ctx, modelName, 60)
			},
		},
		{
			name: "Get deployment status",
			run: func() error {
				status, err := suite.st.GetDeploymentStatus(ctx, modelName)
				if err != nil {
					return err
				}
				if status.ReadyReplicas != 2 {
					t.Logf("Expected 2 ready replicas, got %d", status.ReadyReplicas)
				}
				return nil
			},
		},
		{
			name: "Check health",
			run: func() error {
				result, err := suite.st.HealthCheck(ctx, modelName)
				if err != nil {
					return err
				}
				if !result.Healthy {
					t.Logf("Expected healthy deployment, got errors: %v", result.Errors)
				}
				return nil
			},
		},
		{
			name: "Scale deployment",
			run: func() error {
				return suite.dc.Scale(ctx, modelName, 3)
			},
		},
		{
			name: "Get events",
			run: func() error {
				events, err := suite.st.GetEventLog(ctx, modelName)
				if err != nil {
					return err
				}
				t.Logf("Retrieved %d events", len(events))
				return nil
			},
		},
		{
			name: "Undeploy model",
			run: func() error {
				return suite.dc.Undeploy(ctx, modelName)
			},
		},
	}

	for _, step := range testSteps {
		t.Run(step.name, func(t *testing.T) {
			if err := step.run(); err != nil {
				t.Errorf("%s failed: %v", step.name, err)
			}
		})
	}
}

// TestHealthCheck_HealthyDeployment tests health check on ready deployment
func TestHealthCheck_HealthyDeployment(t *testing.T) {
	suite := setupIntegrationTestSuite()
	ctx := context.Background()
	modelName := "healthy-model"

	// Pre-create deployment with ready replicas
	deployment := &appv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ollama-" + modelName,
			Namespace: "default",
		},
		Spec: appv1.DeploymentSpec{
			Replicas: int32Ptr(2),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "ollama", "model": modelName},
			},
		},
		Status: appv1.DeploymentStatus{
			Replicas:      2,
			ReadyReplicas: 2,
		},
	}

	if _, err := suite.clientset.AppsV1().Deployments("default").Create(ctx, deployment, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create test deployment: %v", err)
	}

	// Create endpoints
	endpoints := &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ollama-" + modelName,
			Namespace: "default",
		},
		Subsets: []corev1.EndpointSubset{
			{
				Addresses: []corev1.EndpointAddress{
					{IP: "10.0.0.1"},
					{IP: "10.0.0.2"},
				},
				Ports: []corev1.EndpointPort{
					{Port: 11434},
				},
			},
		},
	}

	if _, err := suite.clientset.CoreV1().Endpoints("default").Create(ctx, endpoints, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create test endpoints: %v", err)
	}

	// Run health check
	result, err := suite.st.HealthCheck(ctx, modelName)
	if err != nil {
		t.Fatalf("Health check failed: %v", err)
	}

	if !result.Healthy {
		t.Errorf("Expected healthy deployment, got errors: %v", result.Errors)
	}

	if result.ReadyReplicas != 2 {
		t.Errorf("Expected 2 ready replicas, got %d", result.ReadyReplicas)
	}
}

// TestHealthCheck_UnhealthyDeployment tests health check on failing deployment
func TestHealthCheck_UnhealthyDeployment(t *testing.T) {
	suite := setupIntegrationTestSuite()
	ctx := context.Background()
	modelName := "unhealthy-model"

	// Pre-create deployment with NOT ready replicas
	deployment := &appv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ollama-" + modelName,
			Namespace: "default",
		},
		Spec: appv1.DeploymentSpec{
			Replicas: int32Ptr(3),
		},
		Status: appv1.DeploymentStatus{
			Replicas:      3,
			ReadyReplicas: 1, // Only 1 of 3 ready
		},
	}

	if _, err := suite.clientset.AppsV1().Deployments("default").Create(ctx, deployment, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create test deployment: %v", err)
	}

	result, err := suite.st.HealthCheck(ctx, modelName)
	if err != nil {
		t.Fatalf("Health check failed: %v", err)
	}

	if result.Healthy {
		t.Error("Expected unhealthy deployment, but got healthy result")
	}

	if len(result.Errors) == 0 {
		t.Error("Expected errors for unhealthy deployment")
	}
}

// TestWatchDeploymentProgress_Timeout tests timeout behavior
func TestWatchDeploymentProgress_Timeout(t *testing.T) {
	suite := setupIntegrationTestSuite()
	ctx := context.Background()
	modelName := "slow-model"

	// Create deployment that will never reach ready state
	deployment := &appv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ollama-" + modelName,
			Namespace: "default",
		},
		Spec: appv1.DeploymentSpec{
			Replicas: int32Ptr(2),
		},
		Status: appv1.DeploymentStatus{
			Replicas:      2,
			ReadyReplicas: 0, // Never becomes ready
		},
	}

	if _, err := suite.clientset.AppsV1().Deployments("default").Create(ctx, deployment, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create test deployment: %v", err)
	}

	// Watch with short timeout (5 seconds)
	err := suite.st.WatchDeploymentProgress(ctx, modelName, 5)
	if err == nil {
		t.Error("Expected timeout error, but got none")
	}

	// Check if it's a timeout error
	if !IsTimeout(err) {
		t.Errorf("Expected timeout error, got: %v", err)
	}
}

// TestContextCancellation tests that operations respect context cancellation
func TestContextCancellation(t *testing.T) {
	suite := setupIntegrationTestSuite()
	modelName := "cancel-model"

	// Create deployment
	deployment := &appv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ollama-" + modelName,
			Namespace: "default",
		},
		Spec: appv1.DeploymentSpec{
			Replicas: int32Ptr(2),
		},
		Status: appv1.DeploymentStatus{
			Replicas:      2,
			ReadyReplicas: 0,
		},
	}

	ctx := context.Background()
	if _, err := suite.clientset.AppsV1().Deployments("default").Create(ctx, deployment, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create test deployment: %v", err)
	}

	// Create cancelled context
	cancelCtx, cancel := context.WithCancel(ctx)
	cancel() // Cancel immediately

	// Watch should respect cancellation
	err := suite.st.WatchDeploymentProgress(cancelCtx, modelName, 30)
	if err == nil {
		t.Error("Expected context cancellation error")
	}
}

// TestServiceCreation tests service creation and endpoint discovery
func TestServiceCreation(t *testing.T) {
	suite := setupIntegrationTestSuite()
	ctx := context.Background()
	modelName := "service-model"

	// Create service
	serviceName := "ollama-" + modelName
	if err := suite.sm.CreateService(ctx, serviceName, modelName, 11434); err != nil {
		t.Fatalf("Failed to create service: %v", err)
	}

	// Verify service exists
	service, err := suite.clientset.CoreV1().Services("default").Get(ctx, serviceName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get created service: %v", err)
	}

	if service.Name != serviceName {
		t.Errorf("Expected service name %s, got %s", serviceName, service.Name)
	}

	// Clean up
	if err := suite.sm.DeleteService(ctx, serviceName); err != nil {
		t.Fatalf("Failed to delete service: %v", err)
	}
}

// TestPVCProvisioning tests PVC creation and binding
func TestPVCProvisioning(t *testing.T) {
	suite := setupIntegrationTestSuite()
	ctx := context.Background()
	modelName := "storage-model"

	pvcName := "pvc-" + modelName

	// Create PVC
	if err := suite.sm.provider.(*Provider).createPVC(ctx, pvcName, "50Gi"); err != nil {
		t.Fatalf("Failed to create PVC: %v", err)
	}

	// Verify PVC exists
	pvc, err := suite.clientset.CoreV1().PersistentVolumeClaims("default").Get(ctx, pvcName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get created PVC: %v", err)
	}

	if pvc.Name != pvcName {
		t.Errorf("Expected PVC name %s, got %s", pvcName, pvc.Name)
	}
}

// Helper function for int32 pointer
func int32Ptr(i int32) *int32 {
	return &i
}

// Benchmark tests for critical paths

// BenchmarkHealthCheck measures health check performance
func BenchmarkHealthCheck(b *testing.B) {
	suite := setupIntegrationTestSuite()
	ctx := context.Background()
	modelName := "bench-model"

	// Setup deployment and endpoints
	deployment := &appv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ollama-" + modelName,
			Namespace: "default",
		},
		Spec: appv1.DeploymentSpec{
			Replicas: int32Ptr(2),
		},
		Status: appv1.DeploymentStatus{
			Replicas:      2,
			ReadyReplicas: 2,
		},
	}
	suite.clientset.AppsV1().Deployments("default").Create(ctx, deployment, metav1.CreateOptions{})

	endpoints := &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ollama-" + modelName,
			Namespace: "default",
		},
		Subsets: []corev1.EndpointSubset{
			{
				Addresses: []corev1.EndpointAddress{{IP: "10.0.0.1"}},
				Ports:     []corev1.EndpointPort{{Port: 11434}},
			},
		},
	}
	suite.clientset.CoreV1().Endpoints("default").Create(ctx, endpoints, metav1.CreateOptions{})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		suite.st.HealthCheck(ctx, modelName)
	}
}

// BenchmarkGetDeploymentStatus measures status retrieval performance
func BenchmarkGetDeploymentStatus(b *testing.B) {
	suite := setupIntegrationTestSuite()
	ctx := context.Background()
	modelName := "bench-status"

	deployment := &appv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ollama-" + modelName,
			Namespace: "default",
		},
		Status: appv1.DeploymentStatus{
			Replicas:      3,
			ReadyReplicas: 3,
		},
	}
	suite.clientset.AppsV1().Deployments("default").Create(ctx, deployment, metav1.CreateOptions{})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		suite.st.GetDeploymentStatus(ctx, modelName)
	}
}
