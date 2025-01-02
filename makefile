# Makefile

CXX = clang++
CXXFLAGS = -std=c++17 -g -Wall -Wextra -pedantic -I./sciplot

# List all object files
OBJECTS = main.o ForwardNeuralNetwork.o DataPreprocessing.o

# Define the executable target
neural_network: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o neural_network $(OBJECTS)

# Compilation rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

# Clean rule to remove object files and executable
clean:
	rm -f *.o neural_network
