import time
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_loss_curves(results: Dict[str, List[float]], output_directory: str, accuracy_title: str):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
        output_directory (str) : output_directory
    """

    loss = results['train_loss']
    test_loss = results['test_loss']
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    epochs = range(len(results['train_loss']))
    
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('LOSS')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title(accuracy_title)
    plt.xlabel('Epoch')
    plt.legend()

    for i in range(3):
        try:
            plt.savefig(f"{output_directory}/fig_loss_curve_{i}.pdf")

            with open(f"{output_directory}/model_loss_acc_results_{i}.csv", "w") as csvfile:
                for key, value in results.items():
                    value_str = ','.join(map(str, value))
                    csvfile.write(key + "," + value_str + "\n")
            break

        except:
            print(f"\033[0;{'91'}m{f'Attempt {i} -> Can not save loss curve figures and/or spreadsheet summary!'}\033[0m")
            time.sleep(3)

    plt.close()