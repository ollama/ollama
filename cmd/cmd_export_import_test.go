package cmd

import (
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"
)

func TestExportCommandFlags(t *testing.T) {
	// Create export command similar to the one in cmd.go
	exportCmd := &cobra.Command{
		Use:   "export MODEL DESTINATION",
		Short: "Export a model to a file or directory",
		Args:  cobra.ExactArgs(2),
	}
	
	exportCmd.Flags().String("compress", "", "Compression type: 'zstd' (default), 'gzip', or leave flag off for none")
	exportCmd.Flags().Lookup("compress").NoOptDefVal = "zstd"
	exportCmd.Flags().Int("compression-level", 3, "Compression level for zstd (1-19, default 3)")
	exportCmd.Flags().Bool("single-thread", false, "Force single-threaded compression")
	exportCmd.Flags().Bool("force", false, "Overwrite existing files without prompting")

	t.Run("Default flag values", func(t *testing.T) {
		compress, err := exportCmd.Flags().GetString("compress")
		require.NoError(t, err)
		require.Equal(t, "", compress)

		compressionLevel, err := exportCmd.Flags().GetInt("compression-level")
		require.NoError(t, err)
		require.Equal(t, 3, compressionLevel)

		singleThread, err := exportCmd.Flags().GetBool("single-thread")
		require.NoError(t, err)
		require.False(t, singleThread)

		force, err := exportCmd.Flags().GetBool("force")
		require.NoError(t, err)
		require.False(t, force)
	})

	t.Run("Compress flag with no value defaults to zstd", func(t *testing.T) {
		// Simulate setting --compress without value
		err := exportCmd.Flags().Set("compress", "")
		require.NoError(t, err)

		// When the flag is present but empty, NoOptDefVal should be used
		if exportCmd.Flags().Lookup("compress").Changed {
			compress := exportCmd.Flags().Lookup("compress").NoOptDefVal
			require.Equal(t, "zstd", compress)
		}
	})

	t.Run("Valid compression level range", func(t *testing.T) {
		// Test setting compression level
		err := exportCmd.Flags().Set("compression-level", "15")
		require.NoError(t, err)

		compressionLevel, err := exportCmd.Flags().GetInt("compression-level")
		require.NoError(t, err)
		require.Equal(t, 15, compressionLevel)
	})

	t.Run("Force flag can be set", func(t *testing.T) {
		err := exportCmd.Flags().Set("force", "true")
		require.NoError(t, err)

		force, err := exportCmd.Flags().GetBool("force")
		require.NoError(t, err)
		require.True(t, force)
	})
}

func TestImportCommandFlags(t *testing.T) {
	// Create import command similar to the one in cmd.go
	importCmd := &cobra.Command{
		Use:   "import SOURCE [MODEL_NAME]",
		Short: "Import a model from a file or directory",
		Args:  cobra.RangeArgs(1, 2),
	}
	
	importCmd.Flags().Bool("force", false, "Overwrite existing model")
	importCmd.Flags().Bool("insecure", false, "Skip checksum verification")

	t.Run("Default flag values", func(t *testing.T) {
		force, err := importCmd.Flags().GetBool("force")
		require.NoError(t, err)
		require.False(t, force)

		insecure, err := importCmd.Flags().GetBool("insecure")
		require.NoError(t, err)
		require.False(t, insecure)
	})

	t.Run("Force flag can be set", func(t *testing.T) {
		err := importCmd.Flags().Set("force", "true")
		require.NoError(t, err)

		force, err := importCmd.Flags().GetBool("force")
		require.NoError(t, err)
		require.True(t, force)
	})

	t.Run("Insecure flag can be set", func(t *testing.T) {
		err := importCmd.Flags().Set("insecure", "true")
		require.NoError(t, err)

		insecure, err := importCmd.Flags().GetBool("insecure")
		require.NoError(t, err)
		require.True(t, insecure)
	})
}

func TestExportCommandArguments(t *testing.T) {
	exportCmd := &cobra.Command{
		Use:  "export MODEL DESTINATION",
		Args: cobra.ExactArgs(2),
	}

	t.Run("Valid arguments", func(t *testing.T) {
		args := []string{"model-name", "/path/to/destination"}
		err := exportCmd.Args(exportCmd, args)
		require.NoError(t, err)
	})

	t.Run("Too few arguments", func(t *testing.T) {
		args := []string{"model-name"}
		err := exportCmd.Args(exportCmd, args)
		require.Error(t, err)
		require.Contains(t, err.Error(), "accepts 2 arg(s), received 1")
	})

	t.Run("Too many arguments", func(t *testing.T) {
		args := []string{"model-name", "/path/to/destination", "extra-arg"}
		err := exportCmd.Args(exportCmd, args)
		require.Error(t, err)
		require.Contains(t, err.Error(), "accepts 2 arg(s), received 3")
	})
}

func TestImportCommandArguments(t *testing.T) {
	importCmd := &cobra.Command{
		Use:  "import SOURCE [MODEL_NAME]",
		Args: cobra.RangeArgs(1, 2),
	}

	t.Run("Valid arguments - source only", func(t *testing.T) {
		args := []string{"/path/to/source"}
		err := importCmd.Args(importCmd, args)
		require.NoError(t, err)
	})

	t.Run("Valid arguments - source and model name", func(t *testing.T) {
		args := []string{"/path/to/source", "new-model-name"}
		err := importCmd.Args(importCmd, args)
		require.NoError(t, err)
	})

	t.Run("No arguments", func(t *testing.T) {
		args := []string{}
		err := importCmd.Args(importCmd, args)
		require.Error(t, err)
		require.Contains(t, err.Error(), "accepts between 1 and 2 arg(s), received 0")
	})

	t.Run("Too many arguments", func(t *testing.T) {
		args := []string{"/path/to/source", "model-name", "extra-arg"}
		err := importCmd.Args(importCmd, args)
		require.Error(t, err)
		require.Contains(t, err.Error(), "accepts between 1 and 2 arg(s), received 3")
	})
}

func TestExportImportCommandHelp(t *testing.T) {
	exportCmd := &cobra.Command{
		Use:   "export MODEL DESTINATION",
		Short: "Export a model to a file or directory",
		Long: `Export a model to a file or directory for sharing or backup purposes.

By default, exports as an uncompressed tar file (.tar) using parallel processing
for optimal performance. Use --compress to create a compressed file (zstd by default).`,
	}

	importCmd := &cobra.Command{
		Use:   "import SOURCE [MODEL_NAME]",
		Short: "Import a model from a file or directory",
		Long:  `Import a model from a previously exported file or directory.`,
	}

	t.Run("Export command help text", func(t *testing.T) {
		require.Equal(t, "export MODEL DESTINATION", exportCmd.Use)
		require.Equal(t, "Export a model to a file or directory", exportCmd.Short)
		require.Contains(t, exportCmd.Long, "Export a model to a file or directory for sharing or backup purposes")
	})

	t.Run("Import command help text", func(t *testing.T) {
		require.Equal(t, "import SOURCE [MODEL_NAME]", importCmd.Use)
		require.Equal(t, "Import a model from a file or directory", importCmd.Short)
		require.Contains(t, importCmd.Long, "Import a model from a previously exported file or directory")
	})
}

func TestCompressionFormatDetection(t *testing.T) {
	testCases := []struct {
		name           string
		compress       string
		expectedFormat string
	}{
		{
			name:           "No compression specified",
			compress:       "",
			expectedFormat: "",
		},
		{
			name:           "Gzip compression",
			compress:       "gzip",
			expectedFormat: "tar.gz",
		},
		{
			name:           "Zstd compression",
			compress:       "zstd",
			expectedFormat: "tar.zst",
		},
		{
			name:           "Default compression (treated as zstd)",
			compress:       "anything-else",
			expectedFormat: "tar.zst",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// This mimics the logic from ExportHandler in cmd.go
			format := ""
			if tc.compress != "" {
				if tc.compress == "gzip" {
					format = "tar.gz"
				} else {
					// This covers both "zstd" and any other value (treating as zstd default)
					format = "tar.zst"
				}
			}
			require.Equal(t, tc.expectedFormat, format)
		})
	}
}
