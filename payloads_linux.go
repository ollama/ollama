package main

import "embed"

//go:embed build/linux/*
var libEmbed embed.FS
