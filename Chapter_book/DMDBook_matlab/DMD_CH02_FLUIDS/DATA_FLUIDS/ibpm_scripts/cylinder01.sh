#! /usr/bin/bash
./build/ibpm -Re 100 -nx 450 -ny 200 -ngrid 4 -length 9 -xoffset -1 -yoffset -2 -xshift 0.75 -nsteps 500 -restart 500 -tecplot 500 -outdir cylinderLOW/ -geom cylinderLOW/cylinder.geom -dt .02 -ubf 0 -tecplotallgrids 1
