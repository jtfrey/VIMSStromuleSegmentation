#!/usr/bin/env python

from setuptools import setup

setup(  name='VIMSStromuleSegmentation',
        version='1.0',
        description='Stromule segmentation neural net trainor',
        author='Wayne Treible',
        author_email='wtreible@udel.edu',
        url='https://github.com/wtreible/VIMSStromuleSegmentation',
        packages=['VIMSStromuleSegmentation'],
        entry_points = {
            'console_scripts': [
                    'UnetSegmentation=VIMSStromuleSegmentation.job_submit:job_submit_entrypoint',
                    'unet_seg=VIMSStromuleSegmentation.unet_seg:unet_seg_entrypoint',
                ],
            },
        install_requires = [
                'numpy',
                'scikit-image>=0.19',
                'keras>=2.9',
            ],
        zip_safe = True,
    )
