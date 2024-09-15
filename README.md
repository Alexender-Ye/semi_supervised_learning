
# Development of crowdsourcing technology for pavement condition assessment

This repository contains four deep learning models in `h5` format, along with a test dataset for evaluating these models. The test dataset is provided as a compressed file containing two CSV files: `scaled_x_test.csv` and `semi_y_test.csv`.

## Contents

- **Models**: Four deep learning models in `h5` format.
- **Test Dataset**: A compressed file with the following CSV files:
  - `scaled_x_test.csv`: Features for testing (the values of the feature have been scaled).
  - `semi_y_test.csv`: Corresponding labels for testing.

## Usage

### Download the Models and Test Dataset

To get started, download the models and the test dataset from this repository. You can use the following commands:

```bash
# Clone the repository
git clone https://github.com/Alexender-Ye/semi_supervised_learning.git

```

### Load and Test the Models

Below is several examples used to describe how to use the models listed in our paper:


#### script used to test pseudo-label, cGAN-based MLP, and our baseline model
```python
import pandas as pd
from tensorflow.keras.models import load_model

random.seed(42)
tf.random.set_seed(42)
np.random.seed = 42
os.environ['PYTHONHASHSEED'] = str(42)
np.random.RandomState(seed=42)

# Load the test data
x_test = pd.read_csv('scaled_x_test.csv')
y_test = pd.read_csv('semi_y_test.csv')

# Load the model
model = load_model('model_name.h5')

# Evaluate the model
loss, mse = model.evaluate(x_test, y_test)
print(f'MSE: {mse}')
```

#### script used to test mean teacher:
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
import os
import random
from sklearn.metrics import mean_squared_error


random.seed(42)
tf.random.set_seed(42)
np.random.seed = 42
os.environ['PYTHONHASHSEED'] = str(42)
np.random.RandomState(seed=42)

def mean_teacher_loss(z, pre1, pre2, pre3, pre4):
    student_loss = tf.keras.losses.MeanSquaredError()
    supervised_loss = student_loss(z, pre1)
    consistency_loss = 0.2 * float((mean_squared_error(pre1, pre3) + mean_squared_error(pre2, pre4)) / 2)
    return supervised_loss + consistency_loss

# Load the test data
x_test = pd.read_csv('scaled_x_test.csv')
y_test = pd.read_csv('semi_y_test.csv')

# Load the model
model = tf.keras.models.load_model('./mean_teacher_best_model.h5',custom_objects={'mean_teacher_loss': mean_teacher_loss})
pred = model.predict(X_test)
mse = tf.keras.metrics.MeanSquaredError()(Y_test, pred).numpy()
print(f'MSE: {mse}')
```

#### script used to test VIME:
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
import os
import random


random.seed(42)
tf.random.set_seed(42)
np.random.seed = 42
os.environ['PYTHONHASHSEED'] = str(42)
np.random.RandomState(seed=42)

def get_encoder(file_name):
    tmp_model = tf.keras.models.load_model(file_name)
    layer_name = tmp_model.layers[1].name
    layer_output = tmp_model.get_layer(layer_name).output
    encoder = models.Model(inputs=tmp_model.input, outputs=layer_output)
    return encoder

X_test = pd.read_csv("./scaled_x_test.csv")
Y_test = pd.read_csv("./semi_y_test.csv")

file_name = '../vime_encoder_folder'

vime_self_encoder = get_encoder(file_name)
layer_name = vime_self_encoder.layers[1].name
layer_output = vime_self_encoder.get_layer(layer_name).output
encoder = models.Model(inputs=vime_self_encoder.input, outputs=layer_output)
x_test_hat = vime_self_encoder.predict(X_test, verbose=0)

model = tf.keras.models.load_model('./MLP_VIME.h5')
pred = model.predict(x_test_hat)
mse = tf.keras.metrics.MeanSquaredError()(Y_test, pred).numpy()
print(f'MSE: {mse}')

```

### List of Models

1. **MLP_VIME.h5**: Original model enhanced by VIME.
2. **MLP_pseudo**: Original model enhanced by Pseudl-label semi-supervised learning.
3. **MLP.h5**: The original model that serves as a baseline in this work.
4. **mean_teacher_best_model.h5**: Original model enhanced by mean-teacher.
5. **GAN.h5**: Original model enhanced by conditional GAN.

### Testing Dataset Details

- **scaled_x_test.csv**: Contains the feature set for testing the models.
- **semi_y_test.csv**: Contains the labels corresponding to the features in `semi_x_test.csv`.

## Contributing

Contributions are welcome! If you have any improvements or suggestions, feel free to open an issue or submit a pull request.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or need further assistance, please feel free to contact us at [alex.sang@monash.edu](mailto:alex.sang@monash.edu), [qiqin.yu@monash.edu](mailto:qiqin.yu@monash.edu).

## Citation

If you find this repository useful, please consider giving a star ⭐ and citation

```bibtex
@article{sang2024smartphone,
  title={Smartphone-Based IRI Estimation for Pavement Roughness Monitoring: A Data-Driven Approach},
  author={Sang, Ye and Yu, Qiqin and Fang, Yihai and Vo, Viet and Wix, Richard},
  journal={IEEE Internet of Things Journal},
  year={2024},
  publisher={IEEE}
}
