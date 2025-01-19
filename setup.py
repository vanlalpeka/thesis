import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="msc_thesis",
    version="0.1.0",
    author="Vanlal Peka",
    author_email="vanlalpeka88@gmail.com",
    description="An ensemble of shallow submodels for anomaly detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=['msc_thesis', 'msc_thesis.*']),
    install_requires=[
        'imgaug',
        'numpy==1.26',
        'pandas',	
        'scikit-learn',	
        'tensorflow',	
        'openml',	
        'pyod',	
        'tqdm',	
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
