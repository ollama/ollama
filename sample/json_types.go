package sample

import (
	"fmt"
)

type JSONState int

const (
	StateStart JSONState = iota
	StateInObject
	StateInObjectKey
	StateInStructuredKey
	StateInStructuredValue
	StateNewline
	StateTab
	StateSpace
	StateInString
	StateInInt
	StateInFloat
	StateInBool
	StateInNull
	StateInColon
	StateInComma
	StateInTab
	StateInSpaceToValue
	StateInSpaceEndValue
	StateInNewlineEndValue
	StateInObjSpace
	StateInList
	StateInListComma
	StateInValue
	StateInValueEnd
	StateInListEnd
	StateInListObjectEnd
	StateInNewline
	StateInNumber
	StateInNumberEnd
	StateInStringEnd
	StateInObjectKeyEnd
	StateTerminate
	StateInObjectEnd
	StateTransitioningToTerminate
)

var JSONStates = []JSONState{
	StateStart,
	StateInObject,
	StateInObjectKey,
	StateInStructuredKey,
	StateNewline,
	StateTab,
	StateSpace,
	StateInString,
	StateInInt,
	StateInFloat,
	StateInBool,
	StateInNull,
	StateInColon,
	StateInComma,
	StateInTab,
	StateInSpaceToValue,
	StateInSpaceEndValue,
	StateInNewlineEndValue,
	StateInObjSpace,
	StateInList,
	StateInListComma,
	StateInValue,
	StateInValueEnd,
	StateInListEnd,
	StateInListObjectEnd,
	StateInNewline,
	StateInNumber,
	StateInNumberEnd,
	StateInStringEnd,
	StateInObjectKeyEnd,
	StateTerminate,
	StateInObjectEnd,
	StateTransitioningToTerminate,
}

func (s JSONState) String() string {
	switch s {
	case StateStart:
		return "StateStart"
	case StateInObject:
		return "StateInObject"
	case StateInObjectKey:
		return "StateInObjectKey"
	case StateInStructuredKey:
		return "StateInStructuredKey"
	case StateNewline:
		return "StateNewline"
	case StateTab:
		return "StateTab"
	case StateSpace:
		return "StateSpace"
	case StateInString:
		return "StateInString"
	case StateInInt:
		return "StateInInt"
	case StateInFloat:
		return "StateInFloat"
	case StateInBool:
		return "StateInBool"
	case StateInNull:
		return "StateInNull"
	case StateInColon:
		return "StateInColon"
	case StateInComma:
		return "StateInComma"
	case StateInTab:
		return "StateInTab"
	case StateInSpaceToValue:
		return "StateInSpace"
	case StateInObjSpace:
		return "StateInObjSpace"
	case StateInList:
		return "StateInList"
	case StateInListObjectEnd:
		return "StateInListObjectEnd"
	case StateInListComma:
		return "StateInListComma"
	case StateInListEnd:
		return "StateInListEnd"
	case StateInNewline:
		return "StateInNewline"
	case StateInNewlineEndValue:
		return "StateInNewlineEndValue"
	case StateInNumber:
		return "StateInNumber"
	case StateInNumberEnd:
		return "StateInNumberEnd"
	case StateInStringEnd:
		return "StateInStringEnd"
	case StateInObjectKeyEnd:
		return "StateInObjectKeyEnd"
	case StateInSpaceEndValue:
		return "StateInSpaceEndValue"
	case StateTerminate:
		return "StateTerminate"
	case StateInObjectEnd:
		return "StateInObjectEnd"
	default:
		return fmt.Sprintf("Unknown state: %d", s)
	}
}
