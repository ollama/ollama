GOALS := $(or $(MAKECMDGOALS),all)
.PHONY: $(GOALS)
$(GOALS):
	$(MAKE) -C llama $@