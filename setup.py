import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='p2pml',
    version="1.0.0",
    author="Anonymous",
    author_email="Anonymous@gmail.com",
    description="P2PML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": ""
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: LINUX",
    ],
    install_requires=[
        'tensorflow==2.8.0',
        'networkx==2.8',
        'pandas==1.3.3',
        'matplotlib==3.5.0',
        'docker==5.0.3',
        'scipy==1.8.1',
        'fabric==2.7.0',
        'celery==5.2.7'
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8.10",
    entry_points={
        'console_scripts': ['p2pml=p2pml.main:main', 'p2pml-cli=automation:main']
    }

)