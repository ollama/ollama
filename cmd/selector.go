package cmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"golang.org/x/term"

	"github.com/ollama/ollama/api"
)

const maxDisplayedItems = 10

var AppOrder = []string{"claude", "opencode", "droid"}

type SelectItem struct {
	Name        string
	Description string
}

func Select(prompt string, items []SelectItem) (string, error) {
	if len(items) == 0 {
		return "", fmt.Errorf("no items to select from")
	}

	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return "", err
	}
	defer term.Restore(fd, oldState)

	fmt.Fprint(os.Stderr, "\033[?25l")
	defer fmt.Fprint(os.Stderr, "\033[?25h")

	var filter string
	selected := 0
	scrollOffset := 0
	var lastLineCount int

	render := func() {
		filtered := filterItems(items, filter)

		if lastLineCount > 0 {
			fmt.Fprintf(os.Stderr, "\033[%dA", lastLineCount)
		}
		fmt.Fprint(os.Stderr, "\033[J")

		fmt.Fprintf(os.Stderr, "%s %s\r\n", prompt, filter)
		lineCount := 1

		if len(filtered) == 0 {
			fmt.Fprintf(os.Stderr, "  \033[37m(no matches)\033[0m\r\n")
			lineCount++
		} else {
			displayCount := min(len(filtered), maxDisplayedItems)

			for i := 0; i < displayCount; i++ {
				idx := scrollOffset + i
				if idx >= len(filtered) {
					break
				}
				item := filtered[idx]
				if idx == selected {
					if item.Description != "" {
						fmt.Fprintf(os.Stderr, "  \033[1m> %s\033[0m \033[37m- %s\033[0m\r\n", item.Name, item.Description)
					} else {
						fmt.Fprintf(os.Stderr, "  \033[1m> %s\033[0m\r\n", item.Name)
					}
				} else {
					if item.Description != "" {
						fmt.Fprintf(os.Stderr, "    %s \033[37m- %s\033[0m\r\n", item.Name, item.Description)
					} else {
						fmt.Fprintf(os.Stderr, "    %s\r\n", item.Name)
					}
				}
				lineCount++
			}

			if remaining := len(filtered) - scrollOffset - displayCount; remaining > 0 {
				fmt.Fprintf(os.Stderr, "  \033[37m... and %d more\033[0m\r\n", remaining)
				lineCount++
			}
		}

		lastLineCount = lineCount
	}

	clearUI := func() {
		if lastLineCount > 0 {
			fmt.Fprintf(os.Stderr, "\033[%dA", lastLineCount)
			fmt.Fprint(os.Stderr, "\033[J")
		}
	}

	render()

	buf := make([]byte, 3)
	for {
		n, err := os.Stdin.Read(buf)
		if err != nil {
			return "", err
		}

		filtered := filterItems(items, filter)

		switch {
		case n == 1 && buf[0] == 13: // Enter
			if len(filtered) > 0 && selected < len(filtered) {
				clearUI()
				return filtered[selected].Name, nil
			}
		case n == 1 && (buf[0] == 3 || buf[0] == 27): // Ctrl+C or Escape
			clearUI()
			return "", fmt.Errorf("cancelled")
		case n == 1 && buf[0] == 127: // Backspace
			if len(filter) > 0 {
				filter = filter[:len(filter)-1]
				selected = 0
				scrollOffset = 0
			}
		case n == 3 && buf[0] == 27 && buf[1] == 91: // Arrow keys
			if buf[2] == 65 && selected > 0 { // Up
				selected--
				if selected < scrollOffset {
					scrollOffset = selected
				}
			} else if buf[2] == 66 && selected < len(filtered)-1 { // Down
				selected++
				if selected >= scrollOffset+maxDisplayedItems {
					scrollOffset = selected - maxDisplayedItems + 1
				}
			}
		case n == 1 && buf[0] >= 32 && buf[0] < 127: // Printable chars
			filter += string(buf[0])
			selected = 0
			scrollOffset = 0
		}

		render()
	}
}

func filterItems(items []SelectItem, filter string) []SelectItem {
	if filter == "" {
		return items
	}
	var result []SelectItem
	filterLower := strings.ToLower(filter)
	for _, item := range items {
		if strings.Contains(strings.ToLower(item.Name), filterLower) {
			result = append(result, item)
		}
	}
	return result
}

func selectApp() (string, error) {
	var items []SelectItem

	for _, name := range AppOrder {
		app, ok := AppRegistry[name]
		if !ok {
			continue
		}
		items = append(items, SelectItem{Name: app.Name, Description: app.DisplayName})
	}

	if len(items) == 0 {
		return "", fmt.Errorf("no apps available")
	}

	return Select("Select app:", items)
}

func selectConnectedApp() (string, error) {
	connections, err := ListConnections()
	if err != nil {
		return "", err
	}

	if len(connections) == 0 {
		return "", nil
	}

	var items []SelectItem
	for _, conn := range connections {
		app, ok := GetApp(conn.App)
		if !ok {
			continue
		}
		items = append(items, SelectItem{
			Name:        app.Name,
			Description: fmt.Sprintf("%s (%s)", app.DisplayName, conn.Model),
		})
	}

	if len(items) == 0 {
		return "", nil
	}

	return Select("Select app to launch:", items)
}

func confirmLaunch(appName string) (bool, error) {
	return confirmPrompt(fmt.Sprintf("Launch %s now?", appName))
}

func confirmPrompt(prompt string) (bool, error) {
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return false, err
	}
	defer term.Restore(fd, oldState)

	fmt.Fprintf(os.Stderr, "%s [y/n] ", prompt)

	buf := make([]byte, 1)
	for {
		if _, err := os.Stdin.Read(buf); err != nil {
			return false, err
		}

		switch buf[0] {
		case 'Y', 'y', 13:
			fmt.Fprintf(os.Stderr, "yes\r\n")
			return true, nil
		case 'N', 'n', 27, 3:
			fmt.Fprintf(os.Stderr, "no\r\n")
			return false, nil
		}
	}
}

func selectModelForConnect(ctx context.Context, currentModel string) (string, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return "", err
	}

	models, err := client.List(ctx)
	if err != nil {
		return "", err
	}

	if len(models.Models) == 0 {
		return "", fmt.Errorf("no models available. Run 'ollama pull <model>' first")
	}

	var items []SelectItem
	cloudModels := make(map[string]bool)
	for _, m := range models.Models {
		items = append(items, SelectItem{Name: m.Name})
		if m.RemoteModel != "" {
			cloudModels[m.Name] = true
		}
	}

	sort.Slice(items, func(i, j int) bool {
		return strings.ToLower(items[i].Name) < strings.ToLower(items[j].Name)
	})

	if currentModel != "" {
		for i, item := range items {
			if item.Name == currentModel {
				items = append([]SelectItem{item}, append(items[:i], items[i+1:]...)...)
				break
			}
		}
	}

	selected, err := Select("Select model:", items)
	if err != nil {
		return "", err
	}

	if cloudModels[selected] {
		if err := ensureSignedIn(ctx, client); err != nil {
			return "", err
		}
	}

	return selected, nil
}

func ensureSignedIn(ctx context.Context, client *api.Client) error {
	user, err := client.Whoami(ctx)
	if err == nil && user != nil && user.Name != "" {
		return nil
	}

	var aErr api.AuthorizationError
	if !errors.As(err, &aErr) || aErr.SigninURL == "" {
		return err
	}

	yes, err := confirmPrompt("Sign in to ollama.com?")
	if err != nil || !yes {
		return fmt.Errorf("sign in required for cloud models")
	}

	fmt.Fprintf(os.Stderr, "\nTo sign in, navigate to:\n    %s\n\n", aErr.SigninURL)
	fmt.Fprintf(os.Stderr, "\033[90mwaiting for sign in to complete...\033[0m")

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			fmt.Fprintf(os.Stderr, "\n")
			return ctx.Err()
		case <-ticker.C:
			user, err := client.Whoami(ctx)
			if err == nil && user != nil && user.Name != "" {
				fmt.Fprintf(os.Stderr, "\r\033[K\033[A\r\033[K\033[1msigned in:\033[0m %s\n", user.Name)
				return nil
			}
			fmt.Fprintf(os.Stderr, ".")
		}
	}
}
