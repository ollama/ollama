package main

import "github.com/ollama/ollama/api"

var suites = []Suite{
	{
		Name: "basic-qa",
		Tests: []Test{
			{
				Name:   "simple-math",
				Prompt: "What is 2+2? Reply with just the number.",
				Check:  Contains("4"),
			},
			{
				Name:   "capital-city",
				Prompt: "What is the capital of France? Reply with just the city name.",
				Check:  Contains("Paris"),
			},
			{
				Name:   "greeting",
				Prompt: "Say hello",
				Check:  HasResponse(),
			},
		},
	},
	{
		Name: "reasoning",
		Tests: []Test{
			{
				Name:   "logic-puzzle",
				Prompt: "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer yes or no.",
				Check:  Contains("no"),
			},
			{
				Name:   "counting",
				Prompt: "How many letters are in the word 'HELLO'?",
				Check:  Contains("5"),
			},
		},
	},
	{
		Name: "instruction-following",
		Tests: []Test{
			{
				Name:   "json-output",
				Prompt: "Reply with a JSON object containing a 'status' field set to 'ok'.",
				Check:  All(Contains("status"), Contains("ok")),
			},
			{
				Name:   "system-prompt",
				Prompt: "What is your name?",
				System: "You are a helpful assistant named TestBot. When asked your name, always respond with 'TestBot'.",
				Check:  Contains("TestBot"),
			},
		},
	},
	{
		Name: "tool-calling-basic",
		Tests: []Test{
			{
				Name:   "single-tool",
				Prompt: "What's the weather like in San Francisco?",
				Tools:  []api.Tool{weatherTool},
				Check:  CallsTool("get_weather"),
			},
			{
				Name:   "tool-selection",
				Prompt: "What time is it in Tokyo?",
				Tools:  []api.Tool{weatherTool, timeTool},
				Check:  CallsTool("get_time"),
			},
			{
				Name:   "no-tool-needed",
				Prompt: "What is 2+2?",
				Tools:  []api.Tool{weatherTool, timeTool},
				Check:  NoTools(),
			},
		},
	},
	{
		Name: "tool-calling-advanced",
		Tests: []Test{
			{
				Name:   "parallel-calls",
				Prompt: "Get the weather in both New York and Los Angeles.",
				Tools:  []api.Tool{weatherTool},
				Check:  All(CallsTool("get_weather"), MinTools(2)),
			},
			{
				Name:   "multi-param",
				Prompt: "Search for Italian restaurants with prices between $20 and $40.",
				Tools:  []api.Tool{restaurantTool},
				Check:  CallsTool("search_restaurants"),
			},
		},
	},
	{
		Name: "tool-calling-thinking",
		Tests: []Test{
			{
				Name:   "thinking-before-tool",
				Prompt: "I need to know the weather in Paris before I decide what to pack.",
				Tools:  []api.Tool{weatherTool},
				Think:  true,
				Check:  CallsTool("get_weather"),
			},
			{
				Name:   "thinking-multi-tool",
				Prompt: "I'm planning a trip to London. I need to know what time it is there and what the weather is like.",
				Tools:  []api.Tool{weatherTool, timeTool},
				Think:  true,
				Check:  MinTools(1),
			},
		},
	},
}

var weatherTool = api.Tool{
	Type: "function",
	Function: api.ToolFunction{
		Name:        "get_weather",
		Description: "Get the current weather in a given location",
		Parameters: api.ToolFunctionParameters{
			Type:     "object",
			Required: []string{"location"},
			Properties: map[string]api.ToolProperty{
				"location": {
					Type:        api.PropertyType{"string"},
					Description: "The city and state",
				},
			},
		},
	},
}

var timeTool = api.Tool{
	Type: "function",
	Function: api.ToolFunction{
		Name:        "get_time",
		Description: "Get the current time in a timezone",
		Parameters: api.ToolFunctionParameters{
			Type:     "object",
			Required: []string{"timezone"},
			Properties: map[string]api.ToolProperty{
				"timezone": {
					Type:        api.PropertyType{"string"},
					Description: "The timezone name",
				},
			},
		},
	},
}

var restaurantTool = api.Tool{
	Type: "function",
	Function: api.ToolFunction{
		Name:        "search_restaurants",
		Description: "Search for restaurants",
		Parameters: api.ToolFunctionParameters{
			Type:     "object",
			Required: []string{"cuisine"},
			Properties: map[string]api.ToolProperty{
				"cuisine": {
					Type:        api.PropertyType{"string"},
					Description: "Type of cuisine",
				},
				"min_price": {
					Type:        api.PropertyType{"number"},
					Description: "Minimum price",
				},
				"max_price": {
					Type:        api.PropertyType{"number"},
					Description: "Maximum price",
				},
			},
		},
	},
}
