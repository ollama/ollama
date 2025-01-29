package sample

import "github.com/ollama/ollama/model"

type StructuredOutput struct {
	schema *Schema
}

func BuildStructuredOutputGraph(schema *Schema, proc model.TextProcessor) *PDANode {
	// _, stateToNodeMap, err := BuildGraph(proc)
	// if err != nil {
	// 	panic(err)
	// }

	return nil
}

// func constrainGraph(graph *PDANode, schema *Schema) *PDANode {
// 	// If no schema constraints, return original graph node
// 	if schema == nil {
// 		return graph
// 	}

// 	// Create a new node with same state
// 	constrainedNode := NewPDANode(graph.State)

// 	// Copy over existing transitions and masks
// 	constrainedNode.TransitionEdges = make(map[rune]*PDANode)
// 	for r, node := range graph.TransitionEdges {
// 		constrainedNode.TransitionEdges[r] = node
// 	}
// 	constrainedNode.MaskTokenIDToNode = graph.MaskTokenIDToNode

// 	// Apply schema constraints based on type
// 	switch schema.EffectiveType() {
// 	case "object":
// 		// Only allow defined property names in object keys
// 		if graph.State == StateInObjectKey {
// 			// TODO: Add property name validation
// 		}

// 		// Constrain property values based on schema
// 		if graph.State == StateInColon || graph.State == StateInSpace {
// 			// Clear transitions to only allow valid types
// 			constrainedNode.TransitionEdges = make(map[rune]*PDANode)

// 			// Add transitions based on property schemas
// 			for _, prop := range schema.Properties {
// 				switch prop.EffectiveType() {
// 				case "object":
// 					if objNode, ok := graph.TransitionEdges['{']; ok {
// 						constrainedNode.TransitionEdges['{'] = constrainGraph(objNode, prop)
// 					}
// 				case "array":
// 					if arrNode, ok := graph.TransitionEdges['[']; ok {
// 						constrainedNode.TransitionEdges['['] = constrainGraph(arrNode, prop)
// 					}
// 				case "string":
// 					if strNode, ok := graph.TransitionEdges['"']; ok {
// 						constrainedNode.TransitionEdges['"'] = constrainGraph(strNode, prop)
// 					}
// 				case "number":
// 					for _, r := range validNumberRunes {
// 						if numNode, ok := graph.TransitionEdges[r]; ok {
// 							constrainedNode.TransitionEdges[r] = constrainGraph(numNode, prop)
// 						}
// 					}
// 				case "integer":
// 					for _, r := range validIntRunes {
// 						if intNode, ok := graph.TransitionEdges[r]; ok {
// 							constrainedNode.TransitionEdges[r] = constrainGraph(intNode, prop)
// 						}
// 					}
// 				case "boolean":
// 					for _, r := range []rune{'t', 'f'} {
// 						if boolNode, ok := graph.TransitionEdges[r]; ok {
// 							constrainedNode.TransitionEdges[r] = constrainGraph(boolNode, prop)
// 						}
// 					}
// 				case "null":
// 					if nullNode, ok := graph.TransitionEdges['n']; ok {
// 						constrainedNode.TransitionEdges['n'] = constrainGraph(nullNode, prop)
// 					}
// 				}
// 			}
// 		}

// 	case "array":
// 		// Constrain array items based on schema
// 		if schema.Items != nil {
// 			for r, node := range graph.TransitionEdges {
// 				constrainedNode.TransitionEdges[r] = constrainGraph(node, schema.Items)
// 			}
// 		}
// 	}

// 	return constrainedNode
// }
