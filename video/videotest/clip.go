// Package videotest writes small synthetic QuickTime clips for tests, so no
// binary video fixtures are checked in.
package videotest

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"math"
)

// Identity is the track matrix with no transform; a, b, c, d, x and y are
// 16.16 fixed point, u, v and w 2.30.
var Identity = [9]uint32{1 << 16, 0, 0, 0, 1 << 16, 0, 0, 0, 1 << 30}

// Clip describes a QuickTime clip: a red square sliding right across a black
// frame, stored as Photo-JPEG, with an optional 440 Hz tone (on the left
// channel when stereo, the right one silent).
type Clip struct {
	Width, Height int
	Frames, FPS   int
	AudioRate     int // zero for no soundtrack
	Stereo        bool
	Matrix        [9]uint32 // track transform; zero means Identity
}

// BoxSize is the square's side.
func (c Clip) BoxSize() int { return c.Height / 2 }

// BoxAt is where frame i's square starts, before any display rotation.
func (c Clip) BoxAt(i int) int { return i * c.Width / 48 }

func (c Clip) channels() int {
	if c.Stereo {
		return 2
	}
	return 1
}

func (c Clip) frame(i int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, c.Width, c.Height))
	draw.Draw(img, img.Bounds(), image.NewUniform(color.Black), image.Point{}, draw.Src)
	box := image.Rect(c.BoxAt(i), 0, c.BoxAt(i)+c.BoxSize(), c.BoxSize())
	draw.Draw(img, box, image.NewUniform(color.NRGBA{255, 0, 0, 255}), image.Point{}, draw.Src)
	return img
}

// atom is a QuickTime box: a size, a type, and its payload.
func atom(kind string, parts ...[]byte) []byte {
	var body []byte
	for _, p := range parts {
		body = append(body, p...)
	}
	return append(be(uint32(8+len(body)), kind), body...)
}

// be packs uint32, uint16, string and []byte fields big-endian.
func be(fields ...any) []byte {
	var b []byte
	for _, f := range fields {
		switch v := f.(type) {
		case uint32:
			b = binary.BigEndian.AppendUint32(b, v)
		case int:
			b = binary.BigEndian.AppendUint32(b, uint32(v))
		case uint16:
			b = binary.BigEndian.AppendUint16(b, v)
		case string:
			b = append(b, v...)
		case []byte:
			b = append(b, v...)
		}
	}
	return b
}

func matrixBytes(m [9]uint32) []byte {
	var b []byte
	for _, v := range m {
		b = binary.BigEndian.AppendUint32(b, v)
	}
	return b
}

// Bytes encodes the clip with every sample in one chunk per track.
func (c Clip) Bytes() ([]byte, error) {
	if c.Matrix == [9]uint32{} {
		c.Matrix = Identity
	}
	ftyp := atom("ftyp", be("qt  ", 0x200, "qt  "))

	var samples [][]byte
	for i := range c.Frames {
		var b bytes.Buffer
		if err := jpeg.Encode(&b, c.frame(i), &jpeg.Options{Quality: 95}); err != nil {
			return nil, err
		}
		samples = append(samples, b.Bytes())
	}
	var pcm []byte
	audioFrames := 0
	if c.AudioRate > 0 {
		audioFrames = c.AudioRate * c.Frames / c.FPS
		for i := range audioFrames {
			v := int16(8000 * math.Sin(2*math.Pi*440*float64(i)/float64(c.AudioRate)))
			pcm = binary.LittleEndian.AppendUint16(pcm, uint16(v))
			if c.Stereo {
				pcm = binary.LittleEndian.AppendUint16(pcm, 0)
			}
		}
	}

	// The mdat payload starts after ftyp and the mdat header.
	videoOffset := len(ftyp) + 8
	var mdat []byte
	sizes := be(0, 0, len(samples))
	for _, s := range samples {
		mdat = append(mdat, s...)
		sizes = append(sizes, be(len(s))...)
	}
	audioOffset := videoOffset + len(mdat)
	mdat = append(mdat, pcm...)

	const movieScale = 1000
	duration := movieScale * c.Frames / c.FPS
	mvhd := atom("mvhd", be(0, 0, 0, movieScale, duration, 1<<16, uint16(0x100), make([]byte, 10)),
		matrixBytes(Identity), make([]byte, 24), be(3))

	traks := c.track(1, duration, c.FPS, "vide", c.Matrix,
		atom("vmhd", be(1, make([]byte, 8))),
		atom("stsd", be(0, 1), atom("jpeg", make([]byte, 6), be(uint16(1), uint16(0), uint16(0), "appl", 0, 512,
			uint16(c.Width), uint16(c.Height), 72<<16, 72<<16, 0, uint16(1), make([]byte, 32), uint16(24), uint16(0xffff)))),
		atom("stts", be(0, 1, c.Frames, 1)),
		atom("stsc", be(0, 1, 1, c.Frames, 1)),
		atom("stsz", sizes),
		atom("stco", be(0, 1, videoOffset)),
	)
	if c.AudioRate > 0 {
		traks = append(traks, c.track(2, duration, c.AudioRate, "soun", Identity,
			atom("smhd", be(0, 0)),
			atom("stsd", be(0, 1), atom("sowt", make([]byte, 6), be(uint16(1), uint16(0), uint16(0), 0,
				uint16(c.channels()), uint16(16), uint16(0), uint16(0), c.AudioRate<<16))),
			atom("stts", be(0, 1, audioFrames, 1)),
			atom("stsc", be(0, 1, 1, audioFrames, 1)),
			atom("stsz", be(0, 1, audioFrames)),
			atom("stco", be(0, 1, audioOffset)),
		)...)
	}
	return bytes.Join([][]byte{ftyp, atom("mdat", mdat), atom("moov", mvhd, traks)}, nil), nil
}

// track builds a trak atom; media time is counted in scale units, one per
// video frame or audio sample.
func (c Clip) track(id, duration, scale int, handler string, matrix [9]uint32, header []byte, table ...[]byte) []byte {
	var width, height uint32
	volume := uint16(0x100)
	if handler == "vide" {
		width, height, volume = uint32(c.Width)<<16, uint32(c.Height)<<16, 0
	}
	tkhd := atom("tkhd", be(3, 0, 0, id, 0, duration, make([]byte, 8), uint16(0), uint16(0), volume, uint16(0)),
		matrixBytes(matrix), be(width, height))
	mdhd := atom("mdhd", be(0, 0, 0, scale, scale*c.Frames/c.FPS, uint16(0), uint16(0)))
	hdlr := atom("hdlr", be(0, "mhlr", handler, 0, 0, 0, []byte{0}))
	dinf := atom("dinf", atom("dref", be(0, 1), atom("alis", be(1))))
	return atom("trak", tkhd, atom("mdia", mdhd, hdlr, atom("minf", header, dinf, atom("stbl", table...))))
}
