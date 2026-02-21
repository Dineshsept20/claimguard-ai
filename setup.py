"""ClaimGuard AI - Pharmacy Claims Anomaly Detection System."""

from setuptools import setup, find_packages

setup(
    name="claimguard-ai",
    version="0.1.0",
    description="AI-powered pharmacy claims anomaly detection system that identifies "
                "fraud, waste, and abuse patterns in prescription drug claims.",
    author="Dinesh",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "imbalanced-learn>=0.11.0",
        "shap>=0.43.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "streamlit>=1.28.0",
        "faker>=19.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
        ],
    },
)
