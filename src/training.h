#ifndef TRAINING_H
#define TRAINING_H

#include <iostream>
#include <chrono>

#include "matrix.h"
#include "dataset.h"
#include "model.h"

struct TrainingResult {
    std::vector<double> epoch_losses;
    std::vector<double> epoch_times;
    double total_time;
    int convergence_epoch;
    bool converged;
};

void trainImage(Model& model, const Matrix& input, const Matrix& target, int epochs) {
    Matrix normalized_input = input.normalize();
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << std::endl;

        auto [logits, loss] = model.forward(normalized_input, target);
        model.backward(logits, target);
        model.update();

        std::cout << "  Loss: " << loss << std::endl;
    }
}

TrainingResult trainDataset(Model& model, Dataset& dataset, int epochs, double target_loss = 0.1) {
    TrainingResult result;
    auto start_total = std::chrono::high_resolution_clock::now(); // System time start
    result.convergence_epoch = -1; // Convergence epoch reset
    result.converged = false;
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << std::endl;
        double total_loss = 0.0;
        dataset.shuffle();
        auto start_epoch = std::chrono::high_resolution_clock::now(); // Epoch time start

        for (int i = 0; i < dataset.size(); i++) {
            const auto& [input, target] = dataset[i];
            auto [logits, loss] = model.forward(input, target);
            model.backward(logits, target);
            model.update();
            total_loss += loss;
        }

        auto end_epoch = std::chrono::high_resolution_clock::now(); // Epoch time end
        double epoch_time = std::chrono::duration<double>(end_epoch - start_epoch).count();

        double avg_loss = total_loss / dataset.size();
        result.epoch_losses.push_back(avg_loss);
        result.epoch_times.push_back(epoch_time);
        std::cout << "  Avg Loss: " << avg_loss << ", Epoch Time: " << epoch_time << " seconds" << std::endl;

        if (avg_loss < target_loss && result.convergence_epoch == -1) {
            result.convergence_epoch = epoch + 1;
            result.converged = true;
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now(); // System time end
    result.total_time = std::chrono::duration<double>(end_total - start_total).count();
    std::cout << "Total Training Time: " << result.total_time << " seconds" << std::endl;

    if (result.convergence_epoch == -1) {
        std::cout << "No convergence reached within " << epochs << " epochs." << std::endl;
    } else {
        std::cout << "Final Convergence Epoch: " << result.convergence_epoch << std::endl;
    }

    return result;
}

#endif
