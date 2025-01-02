# Compiler and flags
CXX      = clang++
CXXFLAGS = -std=c++17 -O2 -g -pedantic \
           -I./include -I./sciplot

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = .

# Executable name
TARGET = neural_network

# List your source files (only the .cpp filenames, no path)
SOURCES = main.cpp \
          ForwardNeuralNetwork.cpp \
          DataPreprocessing.cpp

# Construct the list of object files in obj/ matching each .cpp in src/
OBJECTS = $(patsubst %.cpp, $(OBJDIR)/%.o, $(SOURCES))

# Default rule: build the executable
all: $(TARGET)

# Link step: create the final binary from object files
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$(TARGET) $(OBJECTS)

# Compilation step: .cpp -> .o
# $< is the source file, $@ is the target .o file
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJDIR)/*.o $(BINDIR)/$(TARGET)
