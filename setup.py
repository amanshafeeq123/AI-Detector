from setuptools import setup, find_packages

setup(
    name="dos_detector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.5.0,<2.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "tensorflow>=2.12.0",
        "matplotlib>=3.5.0,<3.8.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0"
    ],
    python_requires=">=3.8,<3.11",
) 