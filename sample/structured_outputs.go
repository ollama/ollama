package sample

import "github.com/ollama/ollama/model"

type StructuredOutput struct {
	schema         *Schema
	stateToNodeMap map[JSONState]*PDANode
}

func BuildStructuredOutputGraph(schema *Schema, proc model.TextProcessor) *StructuredOutput {
	_, stateToNodeMap, err := BuildGraph(proc)
	if err != nil {
		panic(err)
	}

	return &StructuredOutput{
		schema:         schema,
		stateToNodeMap: stateToNodeMap,
	}
}

func (so *StructuredOutput) schemaToGraph(proc model.TextProcessor) *PDANode {

	schemaType := so.schema.EffectiveType()
	switch schemaType {
	case "object":
		// each prop is a key
		// prevState := StateInObjectKey
		for _, prop := range so.schema.Properties {
			// name of key
			name := prop.Name
			prevState := StateInObjectKey
			for i, r := range name {
				newState := JSONState(int(StateInObjectKey) + i + 1) // Create new unique state for each rune

				// Create new node for this state if it doesn't exist
				if _, exists := so.stateToNodeMap[newState]; !exists {
					so.stateToNodeMap[newState] = &PDANode{
						State:             newState,
						TransitionEdges:   make(map[rune]*PDANode),
						MaskTokenIDToNode: make(map[int32]JSONState),
					}
				}

				// Connect previous state to this state via the rune
				so.stateToNodeMap[prevState].TransitionEdges[r] = so.stateToNodeMap[newState]
				prevState = newState
			}
			// type of value
			// propType := prop.Type
		}
	}
	return nil
}
