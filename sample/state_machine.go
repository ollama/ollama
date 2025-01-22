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
	startToken     token
	endToken       token
	stringToken    token
	objectKeyToken token
	tabToken       token
	spaceToken     token
	newlineToken   token
	newlineSpace   token
	commaToken     token
	commaToken2    token
	commaToken3    token
	colonToken     token
	colonToken2    token
)

func initTokens(proc model.TextProcessor) error {
	var err error
	startToken, err = proc.Encode("{")
	if err != nil {
		return err
	}
	endToken, err = proc.Encode("}")
	if err != nil {
		return err
	}
	stringToken, err = proc.Encode("\"")
	if err != nil {
		return err
	}
	objectKeyToken, err = proc.Encode("\"")
	if err != nil {
		return err
	}
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
	// TODO: figure out how to encode colon correctly
	colonToken, err = proc.Encode("\":")
	if err != nil {
		return err
	}
	fmt.Println("colonToken", colonToken)
	colonToken2, err = proc.Encode(":")
	if err != nil {
		return err
	}
	commaToken, err = proc.Encode(",")
	if err != nil {
		return err
	}
	commaToken2, err = proc.Encode("\",")
	if err != nil {
		return err
	}
	fmt.Println("commaToken2", commaToken2)
	commaToken3, err = proc.Encode("\",\"")
	if err != nil {
		return err
	}
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
	intNode := NewNode(StateInInt)
	commaNode := NewNode(StateInComma)
	colonNode := NewNode(StateInColon)
	stringEndNode := NewNode(StateInStringEnd)
	endNode := NewNode(StateEnd)
	terminateNode := NewNode(StateTerminate)

	sentinelToken := token([]int32{-1})
	intSentinelToken := token([]int32{-2})

	startNode.TransitionEdges[objectNode] = []token{startToken}

	objectNode.TransitionEdges[objectKeyNode] = []token{stringToken}
	// objectNode.TransitionEdges[objectNode] = []token{newlineToken}
	// objectNode.TransitionEdges[objectNode] = []token{spaceToken}

	objectKeyNode.TransitionEdges[objectKeyNode] = []token{sentinelToken}
	objectKeyNode.TransitionEdges[colonNode] = []token{colonToken, colonToken2}
	// characterize end of object key
	objectKeyNode.TransitionEdges[objectKeyEndNode] = []token{stringToken}

	objectKeyEndNode.TransitionEdges[colonNode] = []token{colonToken}

	// objectKeyNode.TransitionEdges[intNode] = []token{sentinelToken}

	intNode.TransitionEdges[intNode] = []token{intSentinelToken}
	intNode.TransitionEdges[commaNode] = []token{commaToken, commaToken2}
	intNode.TransitionEdges[terminateNode] = []token{endToken}

	commaNode.TransitionEdges[objectKeyNode] = []token{newlineToken}

	colonNode.TransitionEdges[stringNode] = []token{stringToken}
	colonNode.TransitionEdges[intNode] = []token{intSentinelToken}

	stringNode.TransitionEdges[stringNode] = []token{sentinelToken}
	stringNode.TransitionEdges[stringEndNode] = []token{stringToken}
	// "\""," Case
	stringNode.TransitionEdges[commaNode] = []token{commaToken2}

	// "\"",\"" Case
	stringNode.TransitionEdges[objectKeyNode] = []token{commaToken3}

	stringEndNode.TransitionEdges[commaNode] = []token{commaToken, commaToken2}
	stringEndNode.TransitionEdges[terminateNode] = []token{endToken}

	endNode.TransitionEdges[terminateNode] = []token{endToken}

	terminateNode.TransitionEdges[terminateNode] = []token{}
	return startNode, nil
}
