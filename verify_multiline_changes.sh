#!/bin/bash

echo "=================================="
echo "Multiline Input Implementation Verification"
echo "=================================="
echo ""

# Check if files were modified
echo "1. Checking modified files..."
if [ -f "readline/readline.go" ]; then
    echo "   ✓ readline/readline.go exists"
else
    echo "   ✗ readline/readline.go not found"
    exit 1
fi

if [ -f "cmd/interactive.go" ]; then
    echo "   ✓ cmd/interactive.go exists"
else
    echo "   ✗ cmd/interactive.go not found"
    exit 1
fi

echo ""
echo "2. Checking for shiftEnterSeq flag..."
if grep -q "var shiftEnterSeq bool" readline/readline.go; then
    echo "   ✓ shiftEnterSeq flag added"
else
    echo "   ✗ shiftEnterSeq flag not found"
fi

echo ""
echo "3. Checking for Shift+Enter detection..."
if grep -q "13;2~" readline/readline.go; then
    echo "   ✓ Windows Terminal Shift+Enter sequence (13;2~) detected"
else
    echo "   ✗ Windows Terminal sequence not found"
fi

if grep -q "27;2;13~" readline/readline.go; then
    echo "   ✓ xterm Shift+Enter sequence (27;2;13~) detected"
else
    echo "   ✗ xterm sequence not found"
fi

echo ""
echo "4. Checking for Alt+Enter support..."
if grep -q "Alt+Enter" readline/readline.go; then
    echo "   ✓ Alt+Enter comment found"
else
    echo "   ✗ Alt+Enter comment not found"
fi

if grep -q "case CharEnter, CharCtrlJ:" readline/readline.go; then
    echo "   ✓ Alt+Enter case handler found"
else
    echo "   ✗ Alt+Enter handler not found"
fi

echo ""
echo "5. Checking help text updates..."
if grep -q "Shift + Enter" cmd/interactive.go; then
    echo "   ✓ Shift+Enter documented in help"
else
    echo "   ✗ Shift+Enter not documented"
fi

if grep -q "Alt + Enter" cmd/interactive.go; then
    echo "   ✓ Alt+Enter documented in help"
else
    echo "   ✗ Alt+Enter not documented"
fi

echo ""
echo "6. Checking for newline insertion..."
if grep -q "buf.Add('\\\\n')" readline/readline.go; then
    echo "   ✓ Newline insertion code found"
else
    echo "   ✗ Newline insertion code not found"
fi

echo ""
echo "=================================="
echo "Verification Complete!"
echo "=================================="
echo ""
echo "To build and test:"
echo "  1. Run: go run . serve"
echo "  2. In another terminal: ./ollama run llama3.2"
echo "  3. Try Shift+Enter or Alt+Enter for multiline input"
echo "  4. Run /? shortcuts to see updated help"
echo ""
