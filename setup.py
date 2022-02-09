import setuptools

setuptools.setup(
        name = 'ponapt',
        packages = setuptools.find_packages(),
        install_requires=[
            'numpy', 'torch', 'tqdm'],)

