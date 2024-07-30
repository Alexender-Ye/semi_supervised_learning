
# Deep Learning Models for Testing

This repository contains four deep learning models in `h5` format, along with a test dataset for evaluating these models. The test dataset is provided as a compressed file containing two CSV files: `semi_x_test.csv` and `semi_y_test.csv`.

## Contents

- **Models**: Four deep learning models in `h5` format.
- **Test Dataset**: A compressed file with the following CSV files:
  - `semi_x_test.csv`: Features for testing.
  - `semi_y_test.csv`: Corresponding labels for testing.

## Usage

### Download the Models and Test Dataset

To get started, download the models and the test dataset from this repository. You can download them directly from the repository's [Releases](https://github.com/Alexender-Ye/semi_supervised_learning.git) page or use the following commands:

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo.git

```

### Load and Test the Models

Below is an example of how to load and test one of the models using Python:

```python
import pandas as pd
from tensorflow.keras.models import load_model

# Load the test data
x_test = pd.read_csv('semi_x_test.csv')
y_test = pd.read_csv('semi_y_test.csv')

# Load the model
model = load_model('your_model_name.h5')

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
```

### List of Models

1. **MLP(VIME).h5**: Original model enhanced by VIME.
2. **MLP(pseudo_label).h5**: Original model enhanced by Pseudl-label semi-supervised learning.
3. **MLP(pure).h5**: The original model that serves as a baseline in this work.
4. **mean_teacher_best_model.h5**: Original model enhanced by mean-teacher.

### Testing Dataset Details

- **semi_x_test.csv**: Contains the feature set for testing the models.
- **semi_y_test.csv**: Contains the labels corresponding to the features in `semi_x_test.csv`.

## Contributing

Contributions are welcome! If you have any improvements or suggestions, feel free to open an issue or submit a pull request.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or need further assistance, please feel free to contact us at [alex.sang@monash.edu](mailto:alex.sang@monash.edu),[qiqin.yu@monash.edu](mailto:qiqin.yu@monash.edu).
