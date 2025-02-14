# MSLearn: Multi-State Surface Learning Package

MSLearn is a Python package designed for learning and training models on multi-state surfaces of molecules using deep learning techniques. 

## Features
- Flexible molecular database recognition
- Various molecular represnetation: xyz, zmatrix, distances, permutation invariant polynomials, graph 
- Supports both feedforward neural network and message-passing neural network representation of multi-state surfaces 
- Permutation invariance of identical nuclei is achieved by either using permutationally invaraint molecular represnetations or as a restrained term in cost function
- Learning of coupled multi-state surfaces by using on-the-fly discovery of diabatic representation or compatible representation 
- Physical properties of surfaces can be achieved by parametrically managed activation function  


## Installation

### Install from Source
Clone the repository and install in editable mode:
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/MSLearn.git
cd MSLearn
pip install -e .
```

### Install from GitHub Directly
```bash
pip install git+https://github.com/shuyinan/MSLearn.git
```

## Usage

### Train a Model
To train a model using a configuration file:
```bash
mslearn-train config.yaml
```
To train a model using a plain text input file:
```bash
mslearn-train input_file
```

Alternatively, you can run:
```bash
python -m MSLearn.main config.yaml
```
or
```bash
python -m MSLearn.main input_file
```

## Configuration
The training pipeline is configured using a YAML file. A sample `config.yaml` file might look like:
```yaml
database: my_data.xyz
database_style: canonical_xyz
molecular_representation: pip
hidden_layers: [256, 128]
architecture: nn
activation: gelu
learning_rate: 0.001
epochs: 100
```

## License
This project is licensed under the MIT License. See `LICENSE.md` for details.

## Contact
For questions or collaboration, reach out to 
Yinan Shu at `yinan.shu.0728@gmail.com`, or 
Donald G. Truhlar at `truhlar@umn.edu`


