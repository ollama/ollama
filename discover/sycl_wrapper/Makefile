CXX = g++
CXXFLAGS = -std=c++17 -fPIC -Wall -Wextra
LDFLAGS = -shared

# SYCL compiler flags - adjust these based on your SYCL implementation
# For Intel oneAPI DPC++
SYCL_FLAGS = -fsycl
SYCL_LIBS = -lsycl

# Enable Intel DPC++ extensions for memory info (if available)
# Comment this out if not using Intel DPC++
INTEL_EXTENSIONS = -DDPCT_COMPATIBILITY_TEMP

# For ComputeCpp
#SYCL_FLAGS = -I$(COMPUTECPP_DIR)/include
#SYCL_LIBS = -L$(COMPUTECPP_DIR)/lib -lComputeCpp

TARGET = sycl_wrapper.so
SOURCES = sycl_wrapper.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SYCL_FLAGS) $(INTEL_EXTENSIONS) -o $@ $^ $(LDFLAGS) $(SYCL_LIBS)

install: $(TARGET)
	mkdir -p $(DESTDIR)/usr/local/lib
	cp $(TARGET) $(DESTDIR)/usr/local/lib/

clean:
	rm -f $(TARGET)

.PHONY: all install clean 