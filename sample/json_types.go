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
	StateInListStartJSON
)

var JSONStates = []JSONState{
	StateStart,
	StateInObject,
	StateInObjectKey,
	StateInStructuredKey,
	StateInStructuredValue,
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
	StateInListStartJSON,
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
	case StateInStructuredValue:
		return "StateInStructuredValue"
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
		return "StateInSpaceToValue"
	case StateInSpaceEndValue:
		return "StateInSpaceEndValue"
	case StateInNewlineEndValue:
		return "StateInNewlineEndValue"
	case StateInObjSpace:
		return "StateInObjSpace"
	case StateInList:
		return "StateInList"
	case StateInListComma:
		return "StateInListComma"
	case StateInValue:
		return "StateInValue"
	case StateInValueEnd:
		return "StateInValueEnd"
	case StateInListEnd:
		return "StateInListEnd"
	case StateInListObjectEnd:
		return "StateInListObjectEnd"
	case StateInNewline:
		return "StateInNewline"
	case StateInNumber:
		return "StateInNumber"
	case StateInNumberEnd:
		return "StateInNumberEnd"
	case StateInStringEnd:
		return "StateInStringEnd"
	case StateInObjectKeyEnd:
		return "StateInObjectKeyEnd"
	case StateTerminate:
		return "StateTerminate"
	case StateInObjectEnd:
		return "StateInObjectEnd"
	case StateTransitioningToTerminate:
		return "StateTransitioningToTerminate"
	case StateInListStartJSON:
		return "StateInListStartJSON"
	default:
		return fmt.Sprintf("Unknown state: %d", s)
	}
}
