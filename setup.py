import setuptools

setuptools.setup(
        name = 'ponapt',
        packages = setuptools.find_packages(),
        install_requires=[
            'numpy', 'torch', 'tqdm'],
        entry_points = {
            'console_scripts':[
                'ponapt-preproc = ponapt.cli.preproc:main',
                'ponapt-train = ponapt.cli.train:main',
                'ponapt-sample = ponapt.cli.sample:main',
                'ponapt-ppl = ponapt.cli.ppl:main',]})

