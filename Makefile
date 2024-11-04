GOALS := $(or $(MAKECMDGOALS),all)
.PHONY: $(GOALS)
$(GOALS):
	@$(MAKE) --no-print-directory -C llama $@