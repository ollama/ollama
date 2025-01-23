package sample

import (
	"fmt"

	"github.com/ollama/ollama/model"
)

type token []int32

type Node struct {
	State           JSONState
	TransitionEdges map[*Node][]token
}

func NewNode(state JSONState) *Node {
	return &Node{
		State:           state,
		TransitionEdges: make(map[*Node][]token),
	}
}

var (
	// startToken             token
	startTokenVariants []token
	// endToken               token
	// stringToken            token
	// objectKeyToken         token
	tabToken     token
	spaceToken   token
	newlineToken token
	newlineSpace token
	// commaToken             token
	// commaToken2            token
	// commaToken3            token
	// colonToken             token
	// colonToken2            token
	colonTokenVariants           []token
	commaTokenVariants           []token
	stringTokenVariants          []token
	endTokenVariants             []token
	objectKeyTokenVariants       []token
	objKeyToColonVariants        []token
	stringToObjectKeyVariants    []token
	stringToCommaVariants        []token
	stringToObjectVariants       []token
	stringEndToObjectEndVariants []token
	stringEndToCommaVariants     []token
)

func ComputeTokenVariants(variants []string, proc model.TextProcessor) ([]token, error) {
	var allTokens token
	for _, variant := range variants {
		if t, err := proc.Encode(variant); err == nil {
			allTokens = append(allTokens, t...)
		}
	}
	if len(allTokens) == 0 {
		return nil, fmt.Errorf("no valid tokens found for variants")
	}
	return []token{allTokens}, nil
}
func initTokens(proc model.TextProcessor) error {
	var err error

	s, err := proc.Decode([]int32{761})
	fmt.Printf("761 decoded %q\n", s)

	// Compute start token variants
	startVariants := []string{"{", " {", "{\n", " {\n"}
	startTokenVariants, err = ComputeTokenVariants(startVariants, proc)
	if err != nil {
		return err
	}
	// Compute end token variants
	endVariants := []string{"}", " }", "}\n", " }\n"}
	endTokenVariants, err = ComputeTokenVariants(endVariants, proc)
	if err != nil {
		return err
	}

	// Compute string token variants
	// TODO: removed \n
	stringVariants := []string{"\"", " \""}
	stringTokenVariants, err = ComputeTokenVariants(stringVariants, proc)
	if err != nil {
		return err
	}
	stringToObjectKeyVariants, err = ComputeTokenVariants([]string{"\",", ",\n", "\",\n"}, proc)
	if err != nil {
		return err
	}
	// objectKeyTokenVariants = []token{stringTokenVariants[0], stringTokenVariants[1]}
	objectKeyTokenVariants = stringTokenVariants
	// Compute whitespace tokens
	tabToken, err = proc.Encode("\t")
	if err != nil {
		return err
	}
	spaceToken, err = proc.Encode(" ")
	if err != nil {
		return err
	}
	newlineToken, err = proc.Encode("\n")
	if err != nil {
		return err
	}
	newlineSpace, err = proc.Encode(" \n")
	if err != nil {
		return err
	}

	// Compute colon variants
	colonVariants := []string{":"}
	colonTokenVariants, err = ComputeTokenVariants(colonVariants, proc)
	if err != nil {
		return err
	}
	objKeyToColonVariants, err = ComputeTokenVariants([]string{"\":"}, proc)
	if err != nil {
		return err
	}

	// Compute comma variants
	commaVariants := []string{",", " ,", ",\n", "\",", "\", "}
	commaTokenVariants, err = ComputeTokenVariants(commaVariants, proc)
	if err != nil {
		return err
	}
	fmt.Printf("commaTokenVariants: %v\n", commaTokenVariants)
	stringToCommaVariants, err = ComputeTokenVariants([]string{"\",", "\","}, proc)
	if err != nil {
		return err
	}

	stringEndToCommaVariants, err = ComputeTokenVariants([]string{",", ",\n"}, proc)
	stringToObjectKeyVariants, err = ComputeTokenVariants([]string{"\",", ",\n", "\","}, proc)
	stringToObjectVariants, err = ComputeTokenVariants([]string{"\",\n"}, proc)
	stringEndToObjectEndVariants, err = ComputeTokenVariants([]string{"\n"}, proc)

	return nil
}

func buildStateMachine(proc model.TextProcessor) (*Node, error) {
	if err := initTokens(proc); err != nil {
		return nil, err
	}

	startNode := NewNode(StateStart)
	objectNode := NewNode(StateInObject)
	objectKeyNode := NewNode(StateInObjectKey)
	objectKeyEndNode := NewNode(StateInObjectKeyEnd)
	stringNode := NewNode(StateInString)
	// intNode := NewNode(StateInInt)
	commaNode := NewNode(StateInComma)
	colonNode := NewNode(StateInColon)
	stringEndNode := NewNode(StateInStringEnd)
	endNode := NewNode(StateEnd)
	terminateNode := NewNode(StateTerminate)

	sentinelToken := token([]int32{-1})
	// intSentinelToken := token([]int32{-2})

	// TODO: cleanup connections of rules
	startNode.TransitionEdges[objectNode] = startTokenVariants

	objectNode.TransitionEdges[objectKeyNode] = stringTokenVariants
	objectNode.TransitionEdges[objectNode] = []token{newlineToken}
	objectNode.TransitionEdges[objectNode] = []token{spaceToken}

	// objectNode.TransitionEdges[objectNode] = []token{newlineToken}
	// objectNode.TransitionEdges[objectNode] = []token{spaceToken}

	objectKeyNode.TransitionEdges[objectKeyNode] = []token{sentinelToken}
	// characterize end of object key
	objectKeyNode.TransitionEdges[objectKeyEndNode] = stringTokenVariants
	objectKeyNode.TransitionEdges[colonNode] = objKeyToColonVariants

	// TODO: enable this - key -> object
	// objectKeyNode.TransitionEdges[objectNode] = startTokenVariants

	// objectKeyNode.TransitionEdges[intNode] = []token{sentinelToken}

	// intNode.TransitionEdges[intNode] = []token{intSentinelToken}
	// intNode.TransitionEdges[commaNode] = commaTokenVariants
	// TODO: handle
	// intNode.TransitionEdges[terminateNode] = endTokenVariants

	commaNode.TransitionEdges[objectKeyNode] = stringTokenVariants
	// commaNode.TransitionEdges[objectNode] = startTokenVariants

	colonNode.TransitionEdges[stringNode] = stringTokenVariants
	//TODO: enable
	// colonNode.TransitionEdges[intNode] = []token{intSentinelToken}
	colonNode.TransitionEdges[objectNode] = startTokenVariants

	stringNode.TransitionEdges[stringNode] = []token{sentinelToken}
	stringNode.TransitionEdges[stringEndNode] = stringTokenVariants
	// TODO: "\""," Case not accounted for
	stringNode.TransitionEdges[commaNode] = stringToCommaVariants

	// TODO: "\"",\"" Case not accounted for
	stringNode.TransitionEdges[objectNode] = stringToObjectVariants

	stringEndNode.TransitionEdges[commaNode] = stringEndToCommaVariants
	stringEndNode.TransitionEdges[objectNode] = stringToObjectKeyVariants
	stringEndNode.TransitionEdges[endNode] = stringEndToObjectEndVariants
	// stringEndNode.TransitionEdges[terminateNode] = endTokenVariants

	// Should be obj end
	// TODO: handle
	endNode.TransitionEdges[terminateNode] = []token{}

	endNode.TransitionEdges[commaNode] = commaTokenVariants

	terminateNode.TransitionEdges[terminateNode] = []token{}
	return startNode, nil
}
