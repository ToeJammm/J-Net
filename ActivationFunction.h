#pragma once
//this interface acts as options for activation functions to use in hidden and output layers
#include <cmath>
#include <memory>

// Abstract Base Class for Activation Functions
class ActivationFunction {
public:
    virtual double activate(double z) const = 0;
    virtual double derivative(double z) const = 0;
    virtual ~ActivationFunction() {}
};

// Sigmoid Activation Function
class Sigmoid : public ActivationFunction {
public:
    double activate(double z) const override {
        return 1.0 / (1.0 + exp(-z));
    }

    double derivative(double z) const override {
        double s = activate(z);
        return s * (1.0 - s);
    }
};

// ReLU Activation Function
class ReLU : public ActivationFunction {
public:
    double activate(double z) const override {
        return z > 0 ? z : 0.0;
    }

    double derivative(double z) const override {
        return z > 0 ? 1.0 : 0.0;
    }
};

// Leaky ReLU Activation Function
class LeakyReLU : public ActivationFunction {
private:
    double alpha; // Slope for z < 0

public:
    LeakyReLU(double alpha = 0.01) : alpha(alpha) {}

    double activate(double z) const override {
        return z > 0 ? z : alpha * z;
    }

    double derivative(double z) const override {
        return z > 0 ? 1.0 : alpha;
    }
};

// Linear Activation Function
class Linear : public ActivationFunction {
public:
    double activate(double z) const override {
        return z;
    }

    double derivative(double z) const override {
        return 1.0;
    }
};
