#include <iostream>
#include <memory>
#include <iomanip>

#include "util.h"
#include "training.h"
#include "model.h"
#include "dataset.h"

int main() {
    int num_conv_layers = 2;
    int epochs = 100;

    std::vector<int> norm_types = {0, 1}; // 0: Normalize, 1: Standardize
    std::vector<double> learning_rates = {0.001, 0.0001};
    std::vector<int> kernel_nums = {1, 2, 4, 8}; // Kernel numbers
    std::vector<int> init_types = {0, 1, 2};  // 0: Set all initial param to 0, 1: He, 2: LeCun
    
    std::ofstream csv_file("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\results\\adam_training_results.csv");
    csv_file << "norm_type,learning_rate,kernel_num,init_type,epoch,avg_loss,epoch_time,total_time,converged,convergence_epoch,accuracy\n";

    // Calculate total number of hyperparameter combinations
    int total_combinations = norm_types.size() * learning_rates.size() * kernel_nums.size() * init_types.size();
    int current_combination = 0;

    for (int norm_type : norm_types) {
        Dataset train_data, test_data;
        train_data.loadCSV("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\data\\mnist_train_1k.csv", 28, 28, 10);
        test_data.loadCSV("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\data\\mnist_test.csv", 28, 28, 10);

        if (norm_type == 0) {
            train_data.normalize_dataset();
            test_data.normalize_dataset();
        }
        else {
            train_data.standardize_dataset();
            test_data.standardize_dataset();
        }
        for (double lr : learning_rates) {
            for (int kernel_num : kernel_nums) {
                for (int init_type : init_types) {
                    current_combination++;
                    double progress_percent = (static_cast<double>(current_combination) / total_combinations) * 100.0;

                    // Print progress for this combination
                    std::cout << "Progress: " << current_combination << "/" << total_combinations 
                              << " (" << std::fixed << std::setprecision(2) << progress_percent << "%) "
                              << "[Norm: " << norm_type << ", LR: " << lr << ", Kernels: " << kernel_num 
                              << ", Init: " << init_type << "]" << std::endl;

                    auto sgd = std::make_unique<SGD>(lr);
                    auto momentum = std::make_unique<Momentum>(lr, 0.9);
                    auto adagrad = std::make_unique<AdaGrad>(lr);
                    auto rmsprop = std::make_unique<RMSProp>(lr, 0.9);
                    auto adam = std::make_unique<Adam>(lr, 0.9, 0.999);

                    std::string optimizer_name = "ADAM";
                    Model model(28, 28, 10, lr, num_conv_layers, std::move(adam), init_type, kernel_num);
                    model.set_training(true);   // Set training mode for batch normalization
                    TrainingResult results = trainDataset(model, train_data, epochs, 0.01);
                    model.set_training(false);  // Set evaluation mode for batch normalization

                    int correct = 0;
                    for (size_t i = 0; i < test_data.size(); i++) {
                        const auto& [input, target] = test_data[i];
                        auto [logits, loss] = model.forward(input, target);
                        Matrix probs = softMax(logits);
                        int guess = argmax(probs);
                        int label = argmax(target);
                        if (guess == label) correct++;
                    }
                    double accuracy = static_cast<double>(correct) / test_data.size();
                    for (size_t epoch = 0; epoch < results.epoch_losses.size(); ++epoch) {
                        csv_file << norm_type << "," << std::fixed << std::setprecision(6) << lr << "," 
                                 << kernel_num << "," << init_type << "," << (epoch + 1) << "," 
                                 << results.epoch_losses[epoch] << "," << results.epoch_times[epoch] << "," 
                                 << results.total_time << "," << (results.converged ? "1" : "0") << "," 
                                 << (results.converged ? results.convergence_epoch : -1) << "," 
                                 << accuracy << "\n";
                    }
                }
            }
        }
    }
    
    csv_file.close();
    std::cout << "Training Completed. Results saved to training_results.csv\n";
    return 0;
}