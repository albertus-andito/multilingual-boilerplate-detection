import setuptools

setuptools.setup(
    name="multilingual-boilerplate-detection",
    version="0.0.1",
    author="Albertus Andito",
    author_email="a.andito@sussex.ac.uk",
    description="Multilingual boilerplate detection model",
    url="https://github.com/albertus-andito/multilingual-boilerplate-detection",
    packages=["m_semtext", "dataset_processor"],
    python_requires=">=3.8",
)