from setuptools import setup, find_packages

setup(
    name="ml-eval-kit",
    version="0.1.0",
    description="Evaluate machine learning models.",
    author="Simon Jorstedt",
    author_email="jorstedtsimon@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib"
    ],
    entry_points={
        "console_scripts": [
            "ml-eval=ml_eval.cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.8',
)
