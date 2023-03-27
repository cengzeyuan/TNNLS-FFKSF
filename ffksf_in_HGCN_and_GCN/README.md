## 2. Setup

### 2.1 Installation with conda

If you don't have conda installed, please install it following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```git clone https://github.com/HazyResearch/hgcn```

```cd hgcn```

```conda env create -f environment.yml```

### 2.2 Installation with pip

Alternatively, if you prefer to install dependencies with pip, please follow the instructions below:

```virtualenv -p [PATH to python3.7 binary] hgcn```

```source hgcn/bin/activate```

```pip install -r requirements.txt```



## 3. Usage

### 3.1 ```set_env.sh```

Before training, run 

```source set_env.sh```

This will create environment variables that are used in the code. 
