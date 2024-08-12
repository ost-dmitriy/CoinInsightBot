from setuptools import setup, find_packages

setup(
    name='CoinInsightBot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'tensorflow',
        'keras',
        'xgboost',
        'statsmodels',
        'pycoingecko',
        'yfinance',
        'python-telegram-bot'
    ],
    entry_points={
        'console_scripts': [
            'crypto-analysis-bot=main:main',
        ],
    },
)
