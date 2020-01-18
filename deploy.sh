m2r README.md
rm -rf dist
python setup.py sdist
twine upload dist/*
