package sample

import (
	"fmt"
	"runtime"
	"time"

	"github.com/ollama/ollama/model"
)

type JSONSampler struct {
	schema        *Schema
	propIdx       int
	propToNodeMap map[string]*PDA
	pdaSampler    *PushdownSampler
	decodedToks   []string
}

func NewJSONSampler(proc model.TextProcessor, schema *Schema) (*JSONSampler, error) {
	if proc == nil {
		return nil, fmt.Errorf("TextProcessor cannot be nil")
	}

	pdaSampler, err := NewPushdownSampler(proc)
	if err != nil {
		return nil, fmt.Errorf("failed to create PushdownSampler: %w", err)
	}

	if schema == nil {
		return &JSONSampler{
			schema:        nil,
			propIdx:       -1,
			propToNodeMap: nil,
			pdaSampler:    pdaSampler,
		}, nil
	}

	// fmt.Println("schema not nil")
	so := &JSONSampler{
		schema:        schema,
		propIdx:       -1,
		propToNodeMap: make(map[string]*PDA),
		pdaSampler:    pdaSampler,
	}

	so.schemaToGraph()

	// Benchmark token decoding
	start := time.Now()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	before := m.Alloc

	vocab := proc.GetVocabulary()
	decodedToks := make([]string, len(vocab.Values))
	for i := range vocab.Values {
		token, err := proc.Decode([]int32{int32(i)})
		if err != nil {
			return nil, err
		}
		decodedToks[i] = token
	}
	so.decodedToks = decodedToks

	runtime.ReadMemStats(&m)
	after := m.Alloc
	fmt.Printf("Token decode memory usage = %.2f MB\n", float64(after-before)/(1024*1024))
	fmt.Printf("Token decode time = %v\n", time.Since(start))

	fmt.Println("--------------------------------")
	fmt.Println("SOSampler")
	fmt.Println("--------------------------------")
	// Benchmark this section
	start = time.Now()
	runtime.ReadMemStats(&m)
	before = m.Alloc

	// TODO: still messed up
	// TODO: recursion use case
	// key masks
	for _, prop := range so.schema.Properties {
		node := so.propToNodeMap[prop.Name]
		// propName -> node
		curState := node.State
		fromNode := node
		so.pdaSampler.CreateMask(fromNode)
		for curState == StateInStructuredKey {
			// there is only one edge
			for r, toNode := range fromNode.TransitionEdges {
				fmt.Println("rune", r, "edge", toNode.State)
				so.pdaSampler.CreateMask(toNode)
				fmt.Printf("created mask for %c\n", r)
				curState = toNode.State
				fmt.Println("next state", curState)
				// TODO: theres an extra gen for " right now
				fromNode = toNode
			}
		}

		if curState != StateInColon {
			return nil, fmt.Errorf("expected state to be StateInColon, got %v", curState)
		}

		// so.pdaSampler.CreateMask(fromNode)

		fromNode = fromNode.TransitionEdges[' ']

		so.pdaSampler.CreateMask(fromNode)
		curState = fromNode.State
		for _, toNode := range fromNode.TransitionEdges {
			fmt.Println("toNode", toNode.State)
		}
	}

	// runtime.ReadMemStats(&m)
	// after = m.Alloc
	// fmt.Printf("Mask creation memory usage = %.2f MB\n", float64(after-before)/(1024*1024))
	// fmt.Printf("Mask creation time = %v\n", time.Since(start))
	// fmt.Println("--------------------------------")

	return so, nil
}

func (s *JSONSampler) schemaToGraph() {
	schemaType := s.schema.EffectiveType()
	switch schemaType {
	case "object":
		// TODO: see if we need to connect these to the JSON graph

		// each prop is a key
		for _, prop := range s.schema.Properties {
			// name of key
			name := prop.Name
			keyNode := &PDA{
				State:             StateInStructuredKey, // this is unchanging, will impact sampling
				TransitionEdges:   make(map[rune]*PDA),
				MaskTokenIDToNode: make(map[int32]*PDA),
			}

			prevNode := keyNode
			for _, r := range name {
				runeNode := &PDA{
					State:             StateInStructuredKey, // this is unchanging, will impact sampling
					TransitionEdges:   make(map[rune]*PDA),
					MaskTokenIDToNode: make(map[int32]*PDA),
				}
				// fmt.Println("runeNode created", runeNode.State)
				// fmt.Printf("runeNode created %c\n", r)

				// since alloc on heap connections wil still map
				prevNode.TransitionEdges[r] = runeNode
				prevNode = runeNode
			}

			// point to end of object key node after all chars are done
			// prevNode.TransitionEdges['"'] = s.pdaSampler.stateToNodeMap[StateInObjectKeyEnd]

			// link to value node
			// Create a node for the end of the key (after the closing quote)
			stringEndNode := &PDA{
				State:             StateInStructuredKey,
				TransitionEdges:   make(map[rune]*PDA),
				MaskTokenIDToNode: make(map[int32]*PDA),
			}
			prevNode.TransitionEdges['"'] = stringEndNode
			prevNode = stringEndNode

			// Add transition for colon after key
			colonNode := &PDA{
				State:             StateInColon,
				TransitionEdges:   make(map[rune]*PDA),
				MaskTokenIDToNode: make(map[int32]*PDA),
			}
			prevNode.TransitionEdges[':'] = colonNode
			prevNode = colonNode

			// Add transition for space after colon
			spaceNode := &PDA{
				State:             StateInSpaceToValue,
				TransitionEdges:   make(map[rune]*PDA),
				MaskTokenIDToNode: make(map[int32]*PDA),
			}
			prevNode.TransitionEdges[' '] = spaceNode
			prevNode = spaceNode

			value := prop.Type
			switch value {
			case "object":
				fmt.Println("object under key: ", name)
			case "array":
				fmt.Println("array under key: ", name)
			case "string":
				fmt.Println("string under key: ", name)
				prevNode.TransitionEdges['"'] = s.pdaSampler.stateToNodeMap[StateInString]
			case "number":
				fmt.Println("number under key: ", name)
				for _, r := range validNumberRunes {
					prevNode.TransitionEdges[r] = s.pdaSampler.stateToNodeMap[StateInNumber]
				}
			case "boolean":
				fmt.Println("boolean under key: ", name)
				prevNode.TransitionEdges['t'] = s.pdaSampler.stateToNodeMap[StateInBool]
				prevNode.TransitionEdges['f'] = s.pdaSampler.stateToNodeMap[StateInBool]
				prevNode.TransitionEdges['n'] = s.pdaSampler.stateToNodeMap[StateInNull]
			}

			// points to start of the key
			s.propToNodeMap[name] = keyNode
			fmt.Println("name", name, "keyNode", keyNode.State)
		}
	}
	// TODO: do values + recursion
}

func (s *JSONSampler) Apply(logits []float64) ([]float64, error) {
	if s.schema == nil {
		return s.pdaSampler.Apply(logits)
	}

	switch s.pdaSampler.curNode.State {
	// TODO: doesnt account for multi rune case
	case StateInObjectKey:
		if s.propIdx > len(s.schema.Properties)-1 {
			return nil, fmt.Errorf("propIdx out of bounds")
		}
		// fmt.Println("in object key - structured outputs")
		// TODO: this tracking should probably be coming from a stack to track nested objects
		// simple case
		s.propIdx++
		fmt.Println("propIdx", s.propIdx)
		prop := s.schema.Properties[s.propIdx]
		fmt.Println("prop", prop.Name)
		s.pdaSampler.curNode = s.propToNodeMap[prop.Name]
		fmt.Println("changed curNode state to", s.pdaSampler.curNode.State)
		logits, err := s.pdaSampler.maskLogits(logits, s.pdaSampler.curNode)
		if err != nil {
			return nil, err
		}
		return logits, nil

	default:

		// Will only happen for the last prop - can also be precomputed.
		if s.propIdx == len(s.schema.Properties)-1 {
			// todo: if i incremenet propidx then i know im in last value as well
			switch s.pdaSampler.curNode.State {
			case StateInObjectEnd:
				fmt.Println("<<<<< in obj end - generating mask for", s.pdaSampler.curNode.State)
				s.pdaSampler.curNode.TransitionEdges = make(map[rune]*PDA)
				s.pdaSampler.curNode = NewPDANode(StateTerminate)
				s.propIdx++

			// TODO: this needs to be optimized in some way, computing mask on the fly is expensive
			case StateInNumber, StateInString, StateInBool, StateInNull, StateInListEnd:
				fmt.Println("<<<<< last prop - generating mask for", s.pdaSampler.curNode.State)
				delete(s.pdaSampler.curNode.TransitionEdges, ',')
				s.pdaSampler.curNode.MaskTokenIDToNode = make(map[int32]*PDA)

				s.pdaSampler.CreateMask(s.pdaSampler.curNode)
				s.propIdx++
			}
		}
		return s.pdaSampler.Apply(logits)
	}
}

func (s *JSONSampler) UpdateState(tokenSlice []int32) ([]int32, error) {
	tokenSlice, err := s.pdaSampler.UpdateState(tokenSlice)
	if err != nil {
		return nil, err
	}

	if s.schema == nil {
		// Don't need to update state for unconstrained JSON sampling
		return tokenSlice, nil
	}

	switch s.pdaSampler.curNode.State {
	case StateInObjectKey:
		s.propIdx++
		fmt.Println("propIdx", s.propIdx)
		prop := s.schema.Properties[s.propIdx]
		fmt.Println("prop", prop.Name)
		s.pdaSampler.curNode = s.propToNodeMap[prop.Name]
		// TODO: this does not work - mike
		// str, err := s.pdaSampler.proc.Decode(tokenSlice)
		// if err != nil {
		// 	return nil, err
		// }
		// fmt.Println("str", str)

		return tokenSlice, nil
	default:
		return tokenSlice, nil
	}
}
