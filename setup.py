from setuptools import setup, find_packages

requirements = [
    'boto3==1.26.104',
    'httplib2==0.21.0',
    'ipywidgets==8.0.2',
    'matplotlib>=3.5.1',
    'numpy>=1.21.5',
    'pandas',
    'plotly',
    'requests',
    'scipy',
    'seaborn',
    'wallstreet',
    'yfinance',
]

setup(
    name='derivslib',
    version='0.0.6',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': []
    },
    description='Provides pricing tools and data for various derivative assets. I am not an attorney, accountant or financial advisor, nor am I holding myself out to be, and the information and tools contained in this package is not a substitute for financial advice from a professional who is aware of the facts and circumstances of your individual situation.',
    url='https://github.com/lwarner100/derivslib',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)