# Laptop Price Prediction

A machine learning-based web application for estimating laptop prices based on hardware specifications, helping students make more informed purchasing decisions.

## Overview

Choosing a laptop can be challenging for university students, particularly when balancing academic requirements, hardware specifications, and financial limitations.

Students at Universitas Gunadarma may face difficulties when determining which laptop to purchase, especially when their desired specifications do not match their available budget.

**Laptop Price Prediction** is a web application developed to help users estimate laptop prices based on selected specifications. The prediction results can serve as an additional reference for comparing laptop configurations, evaluating affordability, and considering purchasing decisions.

## Project Objectives

- Estimate laptop prices based on selected specifications.
- Help students understand the relationship between laptop specifications and estimated prices.
- Support budget considerations when choosing a laptop.
- Provide an additional reference for laptop purchasing decisions.

## Application Preview

### Price Prediction

The application allows users to select laptop specifications and obtain an estimated price.

![Laptop Price Prediction Application](docs/images/prediction-page.png)

## Features

- **Laptop Price Prediction:** Estimate laptop prices based on selected specifications.
- **Specification Input:** Select laptop brands, processors, RAM, storage, screen sizes, operating systems, and GPU specifications.
- **Prediction Results:** Display estimated laptop prices in Indonesian Rupiah (IDR).
- **Input Details:** Display the specifications submitted for prediction.
- **Interactive Web Interface:** Navigate through application pages using Streamlit.

## Dataset

The dataset consists of **646 laptop records** with an initial total of **16 attributes**.

During data preprocessing, relevant laptop specifications were extracted from the product name attribute to obtain more structured information.

After considering their relevance to the target variable (laptop price), **9 attributes were selected** for the prediction process.

### Selected Attributes

| No. | Attribute        | Role            |
| --- | ---------------- | --------------- |
| 1   | Name             | Feature         |
| 2   | VGA              | Feature         |
| 3   | Processor        | Feature         |
| 4   | Storage          | Feature         |
| 5   | Screen Size      | Feature         |
| 6   | RAM              | Feature         |
| 7   | Operating System | Feature         |
| 8   | Brand            | Feature         |
| 9   | Price            | Target Variable |

The selected attributes represent laptop product information and specifications used in the price prediction process.

## Machine Learning Model

The application uses **Multiple Linear Regression** to estimate laptop prices based on the selected laptop specifications.

### Data Splitting

The dataset was divided into training and testing sets using the following ratio:

| Dataset       | Proportion |
| ------------- | ---------: |
| Training Data |        70% |
| Testing Data  |        30% |

### Model Performance

The model evaluation results are as follows:

| Evaluation Metric         |                      Result |
| ------------------------- | --------------------------: |
| R² Score                  |                        0.80 |
| Mean Absolute Error (MAE) | Approximately IDR 2,900,000 |
| Total Dataset             |                 646 records |

### Evaluation Interpretation

- **R² Score (0.80):** The model explains approximately 80% of the variation in laptop prices on the evaluated data.
- **MAE (approximately IDR 2,900,000):** The average absolute difference between predicted and actual laptop prices is approximately IDR 2.9 million.

These metrics provide an overview of the model's performance based on the evaluation conducted during development.

## Technology Stack

| Technology   | Version       | Purpose                                 |
| ------------ | ------------- | --------------------------------------- |
| Python       | Not specified | Main programming language               |
| Streamlit    | 1.45.1        | Interactive web application             |
| Pandas       | 2.2.3         | Data manipulation and input processing  |
| NumPy        | 2.2.5         | Numerical computation                   |
| Seaborn      | 0.13.2        | Data visualization                      |
| Matplotlib   | 3.10.3        | Data visualization                      |
| Scikit-learn | 1.6.1         | Machine learning and data preprocessing |
| Joblib       | 1.5.0         | Model-related serialization support     |

## Application Pages

The application provides four main pages:

### 1. Home

Introduces the application and explains its main features.

### 2. Price Prediction

Allows users to select laptop specifications and obtain estimated prices.

The available inputs include:

- Brand
- RAM capacity
- Processor
- SSD capacity
- Screen size
- Operating system
- GPU brand
- GPU model
- GPU VRAM
- Processor generation
- Processor series

### 3. About Application

Provides information about the technologies, prediction workflow, and dataset used in the application.

### 4. Contact / Feedback

Provides GitHub and email contact information, along with a feedback form interface.

## Prediction Workflow

1. Users select laptop specifications through the Streamlit interface.
2. The selected specifications are organized into a Pandas DataFrame.
3. The machine learning pipeline generates a price prediction.
4. The predicted value is transformed back into its original price scale using the saved scaler.
5. The estimated laptop price is displayed in Indonesian Rupiah (IDR).

## Installation and Usage

### Prerequisites

- Python
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/farismns/prediksi-harga-laptop.git
cd prediksi-harga-laptop
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

**Note:** The application requires the trained model file `model.pkl` in the expected project location. The repository must include this file or provide instructions for obtaining it.

## Prediction Reliability and Data Updates

The reliability of predicted prices depends on the quality and relevance of the dataset used to train the model.

Laptop prices and hardware specifications may change over time due to market conditions, new product releases, and technological developments.

Therefore, the dataset should be updated periodically, followed by model retraining and evaluation, to maintain the model's relevance to current market prices.

If the dataset is not updated, the application can still generate predictions using the existing model. However, its performance on current market prices may differ from the reported evaluation results.

## Disclaimer

The predicted prices are estimates and should be used as supporting references when comparing laptop specifications and considering purchasing budgets.

The application does not guarantee that the predicted price will match the actual market price.

## Contact

- **GitHub:** [farismns](https://github.com/farismns)
- **Email:** [faris.m.saputra@gmail.com](mailto:faris.m.saputra@gmail.com)
