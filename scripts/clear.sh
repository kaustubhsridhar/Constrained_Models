for FOL in Drones
do
    cd ../${FOL}/data
    rm -rf gng_*
    rm -rf voronoi*
    rm -rf bounds_*
    rm -rf saved_models*
    cd ../../scripts
done