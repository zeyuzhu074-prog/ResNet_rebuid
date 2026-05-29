import matplotlib.pyplot as plt
import os

def plot_experiment(history, save_path):

    plt.figure(figsize=(10, 6))

    for model_name, metrics in history.items():

        # train error -> dashed line
        plt.plot(
            metrics['train_error'],
            linestyle='--',
            linewidth=2,
            label=f'{model_name} Train'
        )

        # test error -> solid line
        plt.plot(
            metrics['test_error'],
            linestyle='-',
            linewidth=2,
            label=f'{model_name} Test'
        )

    plt.xlabel('Epoch')

    plt.ylabel('Error (%)')

    plt.title('Experiment Comparison')

    plt.legend()

    plt.grid()

    os.makedirs('./outputs/figures', exist_ok=True)

    plt.savefig(save_path)

    plt.show()

