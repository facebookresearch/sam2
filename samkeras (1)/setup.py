from setuptools import setup, find_packages

setup(
    name='sam2keras',
    version='0.1.0',  # Update with your version
    description='TensorFlow/Keras implementation of the Segment Anything Model 2 (SAM2)',
    author='Iman Jefferson',  # Replace with your name
    author_email='Iman@thepegasusai.com',  # Replace with your email
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.12',  # Use the latest TensorFlow version (adjust if needed)
        'opencv-python',
        'numpy',
        'pygobject',  # For GStreamer video loading
                
    ],
)