#m2r README.md
rm -rf tf_geometric.egg-info
rm -rf dist
python setup.py sdist
twine upload dist/* --verbose
