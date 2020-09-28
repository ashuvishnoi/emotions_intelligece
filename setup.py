from setuptools import setup, find_packages

# setup(include_package_data=True)

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='facial_emotion_recognition',
    include_package_data=True,
    version='0.3.3',
    description='It recognize facial emotions from the image',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(),
    author='Rayyan Akhtar',
    author_email='akhtar.rayyan@live.com',
    install_requires = [
        # "python_version<'3.8'",
        'facenet_pytorch',
        'opencv-python>=3.4.2',
        'numpy>=1.18.1'
    ],
    dependency_links = [
        'https://pypi.org/project/torch/1.4.0/'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
      ]
)


if __name__ == '__main__':
    setup(**setup_args)