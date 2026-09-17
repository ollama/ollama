//go:build integration

package integration

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

const visionOCRReference = "HARBOR-DELTA-8061"

func registerVisionOCRDocumentCases(models []string) {
	registerModelIntegrationCases("vision-ocr-document", models, runVisionOCRDocument)
}

func runVisionOCRDocument(t *testing.T, model string) {
	t.Helper()

	skipUnderMinVRAM(t, 8)
	skipKnownIntegrationFlake(t, "vision-ocr-document", model)
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	requireCapability(ctx, t, client, model, "vision")
	pullOrSkip(ctx, t, client, model)

	document, err := visionOCRDocument()
	if err != nil {
		t.Fatal(err)
	}
	req := api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{
				Role: "user",
				Content: "Scan the entire document and read the unique FINAL AUDIT CODE field. " +
					"Reply with only its value.",
				Images: []api.ImageData{document},
			},
		},
		Stream: &stream,
		Options: map[string]any{
			"seed":        42,
			"temperature": 0.0,
		},
		KeepAlive: &api.Duration{Duration: 10 * time.Second},
	}

	preloadGenerateModel(ctx, t, client, api.GenerateRequest{Model: req.Model})
	skipIfNotGPULoaded(ctx, t, client, req.Model, 80)

	// The shared response normalization collapses whitespace runs but keeps
	// single spaces, and quantized models occasionally split the code with a
	// stray space ("HARBO R-..."). Accept any stream and compare with all
	// whitespace removed instead — every glyph still has to be right.
	msg := DoChat(ctx, t, client, req, []string{""}, 240*time.Second, 30*time.Second)
	if msg == nil {
		return
	}
	despace := func(s string) string { return strings.Join(strings.Fields(s), "") }
	if !strings.Contains(despace(msg.Content), despace(visionOCRReference)) {
		t.Fatalf("%s: audit code %q not found in %q", model, visionOCRReference, msg.Content)
	}
}

func visionOCRDocument() ([]byte, error) {
	const (
		tileWidth  = 800
		tileHeight = 800
		columns    = 3
		rows       = 4
	)

	parsedFont, err := opentype.Parse(goregular.TTF)
	if err != nil {
		return nil, fmt.Errorf("parse document font: %w", err)
	}
	bodyFace, err := opentype.NewFace(parsedFont, &opentype.FaceOptions{
		Size:    40,
		DPI:     72,
		Hinting: font.HintingFull,
	})
	if err != nil {
		return nil, fmt.Errorf("create document font: %w", err)
	}
	referenceFace, err := opentype.NewFace(parsedFont, &opentype.FaceOptions{
		Size:    48,
		DPI:     72,
		Hinting: font.HintingFull,
	})
	if err != nil {
		return nil, fmt.Errorf("create reference font: %w", err)
	}

	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	ink := image.NewUniform(color.RGBA{R: 24, G: 29, B: 36, A: 255})
	rule := color.RGBA{R: 180, G: 186, B: 194, A: 255}
	tile := image.NewRGBA(image.Rect(0, 0, tileWidth, tileHeight))
	draw.Draw(tile, tile.Bounds(), image.NewUniform(white), image.Point{}, draw.Src)
	drawOCRRule(tile, image.Rect(40, 40, tileWidth-40, 44), rule)
	drawOCRRule(tile, image.Rect(40, 130, tileWidth-40, 134), rule)
	drawOCRRule(tile, image.Rect(40, 520, tileWidth-40, 524), rule)
	drawOCRText(tile, bodyFace, ink, 48, 110, "FIELD SERVICE RECORD")
	drawOCRText(tile, bodyFace, ink, 48, 220, "Equipment: turbine assembly")
	drawOCRText(tile, bodyFace, ink, 48, 290, "Inspection: pressure and seals")
	drawOCRText(tile, bodyFace, ink, 48, 360, "Status: passed")
	drawOCRText(tile, bodyFace, ink, 48, 470, "Technician notes: no defects")

	page := image.NewRGBA(image.Rect(0, 0, columns*tileWidth, rows*tileHeight))
	draw.Draw(page, page.Bounds(), image.NewUniform(white), image.Point{}, draw.Src)
	for row := range rows {
		for column := range columns {
			cell := image.NewRGBA(tile.Bounds())
			draw.Draw(cell, cell.Bounds(), tile, image.Point{}, draw.Src)
			drawOCRText(cell, bodyFace, ink, 48, 190, fmt.Sprintf("ROW %d  COLUMN %d", row+1, column+1))

			label := "REFERENCE:"
			reference := fmt.Sprintf("CELL-R%d-C%d-%04d", row+1, column+1, (row+1)*100+column+1)
			if row == rows-1 && column == columns-1 {
				label = "FINAL AUDIT CODE:"
				reference = visionOCRReference
			}
			drawOCRText(cell, referenceFace, ink, 48, 640, label)
			drawOCRText(cell, referenceFace, ink, 48, 710, reference)

			origin := image.Pt(column*tileWidth, row*tileHeight)
			draw.Draw(page, image.Rectangle{Min: origin, Max: origin.Add(cell.Bounds().Size())}, cell, image.Point{}, draw.Src)
		}
	}

	var encoded bytes.Buffer
	if err := png.Encode(&encoded, page); err != nil {
		return nil, fmt.Errorf("encode OCR document: %w", err)
	}
	return encoded.Bytes(), nil
}

func drawOCRText(dst draw.Image, face font.Face, source image.Image, x, y int, text string) {
	drawer := font.Drawer{
		Dst:  dst,
		Src:  source,
		Face: face,
		Dot:  fixed.P(x, y),
	}
	drawer.DrawString(text)
}

func drawOCRRule(dst draw.Image, bounds image.Rectangle, fill color.Color) {
	draw.Draw(dst, bounds, image.NewUniform(fill), image.Point{}, draw.Src)
}
