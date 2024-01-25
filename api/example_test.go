package api_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/server"
)

const testModelName = "llama-2-tiny-random"
const testGgufFilePath = "data/llama-2-tiny-random.gguf"
const testModelFilePath = "data/llama-2-tiny-random.model"

func TestMain(m *testing.M) {

	// Here we start a server in the background to run with our tests.
	host, port, err := net.SplitHostPort(os.Getenv("OLLAMA_HOST"))
	if err != nil {
		host, port = "127.0.0.1", "11434"
		if ip := net.ParseIP(strings.Trim(os.Getenv("OLLAMA_HOST"), "[]")); ip != nil {
			host = ip.String()
		}
	}

	ln, err := net.Listen("tcp", net.JoinHostPort(host, port))
	if err != nil {
		log.Fatal(err)
	}

	go server.Serve(ln)

	// Create a new context
	ctx := context.Background()

	// Create a new client
	client, err := api.ClientFromEnvironment()

	// load our test model
	err = loadTestModel(ctx, client, testModelName)
	if err != nil {
		log.Fatal(err)
	}
	defer deleteTestModel(ctx, client, testModelName) // not sure if we need this and is more of a question of what developers would prefer.

	result := m.Run()

	ln.Close()

	os.Exit(result)
}

func ExampleClient_Heartbeat() {
	// Create a new context
	ctx := context.Background()

	// Create a new client
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	// Get a heartbeat from the server. If no error is returned, we know the server is up.
	err = client.Heartbeat(ctx)
	if err != nil {
		log.Fatal(err)
	}

	if err == nil {
		fmt.Println("server is up")
	}
	// Output:
	// server is up
}

func ExampleClient_Version() {
	// Create a new context
	ctx := context.Background()

	// Create a new client
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	// Get the version of the server
	version, err := client.Version(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// if we get an empty version, something is wrong
	if version == "" {
		log.Fatal("version is empty")
	} else {
		fmt.Println("version is not empty")
	}
	// Output:
	// version is not empty
}

func ExampleClient_List() {

	// Create a new context
	ctx := context.Background()

	// Create a new client
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	// List all the models on the server as tags
	tags, err := client.List(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// if we get an empty list it means that there are no models on the server
	if len(tags.Models) == 0 {
		log.Fatal("no tags returned")
	} else {
		fmt.Println("tags returned")
	}
	// Output:
	// tags returned
}

func ExampleClient_Chat() {

	// Create a new context
	ctx := context.Background()

	// Create a new client
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	// List all the models on the server as tags
	tags, err := client.List(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// if we get an empty list it means that there are no models on the server
	if len(tags.Models) == 0 {
		log.Fatal("no tags returned")
	}

	// sort the tags.models by Size least to greatest
	sort.Slice(tags.Models, func(i, j int) bool {
		return tags.Models[i].Size < tags.Models[j].Size
	})

	// Get the first (smallest) model in the list (this will like the defined in the testModelName constant)
	model := tags.Models[0].Model

	// Now we're

	// Create a new chat message to send to the server. Role must be defined and can be "user", system", or a third one that I forget.
	// These should be defined in the api package as an enum constant or something.
	chatMessage := api.Message{
		Content: "Hello! Tell me about yourself in exactly one sentence.",
		Role:    "user",
	}

	// Create a new chat request to send to the server. This specifies the model and the messages to send.
	req := &api.ChatRequest{
		Model:    model,
		Messages: []api.Message{chatMessage},
	}

	// Create a function to handle the response from the server. This is a callback function that will be called for each response from the server.
	var modelResponseText string                  // we will store the response from the server in this variable
	fn := func(response api.ChatResponse) error { // this callback function fires for each token returned from the server

		modelResponseText += response.Message.Content // append the response to the modelResponseText variable to retrieve the whole message
		return nil
	}

	if err := client.Chat(ctx, req, fn); err != nil { // send the request to the server and if there is an error, log it
		if errors.Is(err, context.Canceled) {
			log.Fatal("context was canceled")
		}
		log.Fatal(err)
	}

	if modelResponseText == "" {
		log.Fatal("modelResponseText is empty")
	} else {
		fmt.Println("model responded")
	}

	// Output:
	// model responded
}

func ExampleClient_Create() {

	// Create a new context
	ctx := context.Background()

	// Create a new client
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	ggufFilePath := testGgufFilePath
	modelFilePath := testModelFilePath
	modelName := "tiny-spoof"

	// get the model file bytes from the testModel file
	modelBytes, err := os.ReadFile(modelFilePath)
	if err != nil {
		log.Fatal(err)
	}

	modelString := string(modelBytes)

	// Create a new create request
	req := &api.CreateRequest{
		Path:      ggufFilePath,
		Model:     modelName,
		Modelfile: modelString,
	}

	var progressMessages []string // we will store the response from the server in this variable
	// Create a function to handle the response from the server. This is a callback function that will be called for each response from the server.
	fn := func(response api.ProgressResponse) error { // this callback function fires for each response returned from the server

		progressMessages = append(progressMessages, response.Status) // append the response to the modelResponseText variable to retrieve the whole message
		return nil
	}

	// Create the model
	if err := client.Create(ctx, req, fn); err != nil { // send the request to the server and if there is an error, log it
		if errors.Is(err, context.Canceled) {
			log.Fatal("context was canceled")
		}
		log.Fatal(err)
	}

	// List all models on the server
	tags, err := client.List(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// Iterate through the tags and check if the model was created
	modelCreatedFlag := false
	for _, tag := range tags.Models {
		if strings.Contains(tag.Model, modelName) {
			modelCreatedFlag = true
		}
	}

	// delete the model
	deleteRequest := &api.DeleteRequest{
		Model: modelName,
	}

	if err := client.Delete(ctx, deleteRequest); err != nil { // send the request to the server and if there is an error, log it
		if errors.Is(err, context.Canceled) {
			log.Fatal("context was canceled")
		}
		log.Fatal(err)
	}

	// check if the model was deleted
	tags, err = client.List(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// iterate through the tags and check if the model was deleted
	for _, tag := range tags.Models {
		if strings.Contains(tag.Model, modelName) {
			log.Fatal("model was not deleted")
		}
	}

	if modelCreatedFlag == true {
		fmt.Println("model created")
	} else {
		log.Fatal("model not created")
	}
	// Output:
	// model created
}

func ExampleClient_Show() {

	// Create a new context
	ctx := context.Background()

	// Create a new client
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	// List all the models on the server as tags
	tags, err := client.List(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// if we get an empty list it means that there are no models on the server
	if len(tags.Models) == 0 {
		log.Fatal("no tags returned")
	}

	// Get the first model in the list
	model := tags.Models[0].Model

	request := api.ShowRequest{
		Model: model,
	}

	// Show the model
	show, err := client.Show(ctx, &request)

	// if we get an empty show, something is wrong
	if show == nil {
		log.Fatal("show is empty")
	} else {
		fmt.Println("show is not empty")
	}
	// Output:
	// show is not empty
}

func loadTestModel(ctx context.Context, client *api.Client, modelName string) error {

	ggufFilePath := testGgufFilePath
	modelFilePath := testModelFilePath

	// get the model file bytes from the testModel file
	modelBytes, err := os.ReadFile(modelFilePath)
	if err != nil {
		return err
	}

	modelString := string(modelBytes)

	// Create a new create request
	req := &api.CreateRequest{
		Path:      ggufFilePath,
		Model:     modelName,
		Modelfile: modelString,
	}

	var progressMessages []string // we will store the response from the server in this variable
	// Create a function to handle the response from the server. This is a callback function that will be called for each response from the server.
	fn := func(response api.ProgressResponse) error { // this callback function fires for each response returned from the server

		progressMessages = append(progressMessages, response.Status) // append the response to the modelResponseText variable to retrieve the whole message
		return nil
	}

	// Create the model
	if err := client.Create(ctx, req, fn); err != nil { // send the request to the server and if there is an error, log it
		if errors.Is(err, context.Canceled) {
			log.Fatal("context was canceled")
		}
		return err
	}

	return nil
}

func deleteTestModel(ctx context.Context, client *api.Client, modelName string) error {

	deleteRequest := &api.DeleteRequest{
		Model: modelName,
	}

	if err := client.Delete(ctx, deleteRequest); err != nil { // send the request to the server and if there is an error, log it
		if errors.Is(err, context.Canceled) {
			log.Fatal("context was canceled")
		}
		return err
	}

	return nil
}
