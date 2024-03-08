package main

import (
	"flag"
	"fmt"
	"reflect"
	"strings"
)

// parseArgs parses the provided args using a flag.FlagSet that is
// dynamically build using reflection for the provided type. The type fields
// that have a "flag" tag are used to build the flags. The flag tag should
// include either a ('-'). Example usage:
//
//	func main() {
//		var flags struct {
//			Modelfile string `flag:"f,path to the Modelfile"`
//		}
//
//		fs := readFlags(os.Args[1:], &flags)
//		fs.Parse(os.Args[1:])
//	}
func readFlags(name string, args []string, v any) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ExitOnError)
	defer fs.Parse(args)
	if v == nil {
		return fs
	}

	for i := 0; i < reflect.ValueOf(v).NumField(); i++ {
		f := reflect.ValueOf(v).Field(i)
		if !f.CanSet() {
			continue
		}

		tag := f.Type().Field(i).Tag.Get("flag")
		if tag == "" {
			continue
		}
		var name, usage string
		if i := strings.Index(tag, ","); i != -1 {
			name = tag[:i]
			usage = tag[i+1:]
		} else {
			name = tag
		}

		// TODO(bmizerany): add more types as needed
		switch f.Kind() {
		case reflect.String:
			fs.StringVar(f.Addr().Interface().(*string), name, "", usage)
		case reflect.Bool:
			fs.BoolVar(f.Addr().Interface().(*bool), name, false, usage)
		default:
			panic(fmt.Sprintf("unsupported type %v", f.Kind()))
		}
	}
	return fs
}
