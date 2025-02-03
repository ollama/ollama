package sample

import (
	"fmt"
	"runtime"
	"time"

	"github.com/ollama/ollama/model"
)

type SOSampler struct {
	schema        *Schema
	propIdx       int
	propToNodeMap map[string]*PDANode
	pdaSampler    *PushdownSampler
}

func NewSOSampler(schema *Schema, proc model.TextProcessor) (*SOSampler, error) {
	pdaSampler := NewPushdownSampler(proc)

	so := &SOSampler{
		schema:        schema,
		propIdx:       -1,
		propToNodeMap: make(map[string]*PDANode),
		pdaSampler:    pdaSampler,
	}

	so.schemaToGraph()

	vocab := proc.GetVocabulary()
	decodedToks := make([]string, len(vocab.Values))
	for i := range vocab.Values {
		token, err := proc.Decode([]int32{int32(i)})
		if err != nil {
			return nil, err
		}
		decodedToks[i] = token
	}

	fmt.Println("--------------------------------")
	fmt.Println("SOSampler")
	fmt.Println("--------------------------------")
	// Benchmark this section
	start := time.Now()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	before := m.Alloc

	// TODO: still messed up
	for _, node := range so.propToNodeMap {
		// propName -> node
		curState := node.State
		fromNode := node
		CreateMask(fromNode, proc, decodedToks, vocab)
		for curState == StateInStructuredKey {
			// there is only one edge
			for r, toNode := range fromNode.TransitionEdges {
				// fmt.Println("rune", r, "edge", toNode.State)
				CreateMask(toNode, proc, decodedToks, vocab)
				fmt.Printf("created mask for %c\n", r)
				curState = toNode.State
				fmt.Println("next state", curState)
				// TODO: theres an extra gen for " right now
				fromNode = toNode
			}
		}
	}

	runtime.ReadMemStats(&m)
	after := m.Alloc
	fmt.Printf("Mask creation memory usage = %.2f MB\n", float64(after-before)/(1024*1024))
	fmt.Printf("Mask creation time = %v\n", time.Since(start))
	fmt.Println("--------------------------------")

	return so, nil
}

func (s *SOSampler) schemaToGraph() {
	schemaType := s.schema.EffectiveType()
	switch schemaType {
	case "object":
		// TODO: see if we need to connect these to the JSON graph
		// prevState := StateInObjectKey
		// prevNode := so.stateToNodeMap[prevState]

		// each prop is a key
		for _, prop := range s.schema.Properties {
			// name of key
			name := prop.Name
			// prevState := StateInObjectKey
			keyNode := &PDANode{
				State:             StateInStructuredKey, // this is unchanging, will impact sampling
				TransitionEdges:   make(map[rune]*PDANode),
				MaskTokenIDToNode: make(map[int32]*PDANode),
			}

			prevNode := keyNode
			for _, r := range name {
				runeNode := &PDANode{
					State:             StateInStructuredKey, // this is unchanging, will impact sampling
					TransitionEdges:   make(map[rune]*PDANode),
					MaskTokenIDToNode: make(map[int32]*PDANode),
				}
				fmt.Println("runeNode created", runeNode.State)
				fmt.Printf("runeNode created %c\n", r)
				// since alloc on heap connections wil still map
				prevNode.TransitionEdges[r] = runeNode
				prevNode = runeNode
			}
			// point to end of object key node after all chars are done
			prevNode.TransitionEdges['"'] = s.pdaSampler.stateToNodeMap[StateInObjectKeyEnd]
			// points to start of the key
			s.propToNodeMap[name] = keyNode
			fmt.Println("name", name, "keyNode", keyNode.State)
		}
	}
}

func (s *SOSampler) Sample(logits []float64) ([]float64, error) {
	switch s.pdaSampler.curNode.State {
	// doesnt account for multi rune case
	case StateInObjectKey:
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
		return s.pdaSampler.Sample(logits)
	}

}

func (s *SOSampler) UpdateState(tokenSlice []int32) error {
	return s.pdaSampler.UpdateState(tokenSlice)
}
