from setuptools import find_packages, setup

__version__ = "1.0.0"

# Load README
with open('README.md', encoding = 'utf-8') as f:
    long_description = f.read()

setup(
    name = 'IDSL_MINT',
    version = '1.0.0',
    author = 'Sadjad Fakouri Baygi, Dinesh Barupal',
    author_email = 'sadjad.fakouri-baygi@mssm.edu, dinesh.barupal@mssm.edu',
    description = 'MS/MS interpretation by ensemble of deep learning models',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/idslme/IDSL_MINT',
    project_urls = {
        'Source': 'https://github.com/idslme/IDSL_MINT'
    },
    license = 'MIT',
    packages = find_packages(),
    entry_points = {
        'console_scripts':
            ['MINT_workflow=IDSL_MINT.IDSL_MINT_workflow:MINT_workflow']
    },
    install_requires = [
	'rdkit',
        'pathlib>=1.0.1',
        'matplotlib>=3.1.3',
        'numpy>=1.18.1',
        'torch>=2.0',
        'torchinfo>=1.8.0',
        'tqdm>=4.45.0',
        'PyYAML',
        'joblib',
        'pandas',
        'openpyxl'
    ],
    python_requires = '>=3.7',
    classifiers = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Linux'
    ],
    keywords = [
        'chemistry',
        'mass spectrometry',
        'msp',
        'spectral entropy',
        'large language model',
        'molecular finger print',
        'SMILES'
    ]
)